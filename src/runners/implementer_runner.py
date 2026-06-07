"""
Local SWE-bench ImplementerAgent runner.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, List, Optional

from base.engine.logs import logger
from benchmark.common.env import Environment
from benchmark.common.runner import Runner, StepRecord, LevelResult
from src.verifier import verify_notes, verify_notes_llm, parse_invalid_evidence_count

NO_PROGRESS_LIMIT = 8
PATCH_READY_AUTO_FINISH_AFTER = None

def _action_signature(action: Any) -> str:
    """
    Stable signature of an action for loop detection. Keyed on the executed
    command (or action type + scalar params), deliberately EXCLUDING
    'reasoning' and 'notes' so near-identical prose doesn't mask a command that
    is in fact byte-identical.
    """
    if not isinstance(action, dict):
        return str(action)[:200]
    a = action.get("action", "")
    p = action.get("params", {}) or {}
    if isinstance(p, dict) and "command" in p:
        return f"{a}:{(p.get('command') or '').strip()}"
    if isinstance(p, dict) and a == "error":
        return f"error:{(p.get('message') or '').strip()[:120]}"
    if isinstance(p, dict):
        scalars = {k: v for k, v in p.items()
                   if k not in ("reasoning", "notes") and isinstance(v, (str, int, float, bool))}
        return f"{a}:{sorted(scalars.items())}"
    return f"{a}:{str(p)[:160]}"

def _source_like_path(path: str) -> bool:
    """True when a diff path looks like a candidate source edit.

    This intentionally uses only generic path categories: test trees,
    temp/cache/build trees, and generated/binary suffixes.
    """
    p = (path or "").strip().replace("\\", "/")
    if not p:
        return False
    parts = p.split("/")
    lowered = [x.lower() for x in parts]
    generated_dirs = {
        ".cache",
        ".doctrees",
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "__pycache__",
        "_build",
        "build",
        "dist",
        "node_modules",
        "target",
    }
    if any(x in generated_dirs for x in lowered):
        return False
    if lowered[0] in {"tmp", "temp"}:
        return False
    if any(x in {"tests", "test", "testing"} for x in lowered):
        return False
    base = lowered[-1]
    generated_suffixes = (
        ".class",
        ".doctree",
        ".gif",
        ".gz",
        ".jar",
        ".jpeg",
        ".jpg",
        ".o",
        ".pdf",
        ".pickle",
        ".png",
        ".pyc",
        ".pyo",
        ".so",
        ".tar",
        ".tgz",
        ".zip",
    )
    if base.endswith(generated_suffixes):
        return False
    return True

async def _current_source_diff_files(env: Environment) -> List[str]:
    """Best-effort list of current source-like working-tree diffs."""
    executor = getattr(env, "_executor", None)
    if executor is None or not hasattr(executor, "execute_command"):
        return []
    try:
        output, code = await executor.execute_command(
            "cd /testbed && git -c core.fileMode=false diff --name-only HEAD -- .",
            timeout=20,
        )
    except TypeError:
        output, code = await executor.execute_command(
            "cd /testbed && git -c core.fileMode=false diff --name-only HEAD -- ."
        )
    except Exception:
        return []
    if code not in (0, None):
        return []
    files = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    return [p for p in files if _source_like_path(p)]

def _patch_ready_block_reason_from_notes(notes: List[Any]) -> str:
    """Detect explicit local failure notes after a patch-ready checkpoint."""
    for note in notes or []:
        if not isinstance(note, dict):
            continue
        type_name = str(note.get("type", "")).upper()
        if type_name == "FAIL":
            content = " ".join(str(note.get("content", "")).split())
            if content:
                return f"local FAIL after patch-ready checkpoint: {content[:160]}"
            return "local FAIL after patch-ready checkpoint"
    return ""

class SWEBenchImplementerRunner(Runner):
    """Runner for ImplementerAgent (standard agent-environment loop) for SWE-bench."""

    exec_timeout: Optional[float] = None
    patch_ready_checkpoint_enabled: bool = False
    patch_summary_finish_handoff_enabled: bool = False
    agent_stop_perf: Optional[float] = None
    note_verifier_mode: str = "deterministic"

    async def run(self, agent, env: Environment) -> LevelResult:
        """Run ImplementerAgent with standard interaction loop."""
        import inspect
        from base.engine.logs import LogLevel

        logger.info(f"[SWEBenchImplementerRunner] Starting ImplementerAgent execution")

        try:
            info = env.get_basic_info()
            agent.reset(info)

            env._done = False
            env._steps = 0
            env._command_history = []
            state_info = env._aci_manager.state.to_status_line()
            obs = {
                "message": "Continuing task. Repository state preserved from prior ImplementerAgent.",
                "output": (
                    f"Environment ready (preserved state).\n{state_info}\n\n"
                    "Note: any edits made by previous ImplementerAgents in this task are still in place."
                ),
                "state_info": state_info,
                "current_step": 0,
                "max_steps": getattr(getattr(env, "config", None), "max_steps", info.max_steps),
                "repo": getattr(getattr(env, "instance", None), "repo", None),
            }

            history = []
            total_reward = 0.0
            max_steps = info.max_steps
            recent_sigs: List[str] = []  # M6: rolling action signatures
            patch_ready_enabled = bool(getattr(self, "patch_ready_checkpoint_enabled", False))
            patch_ready_step: Optional[int] = None
            patch_ready_summary = ""
            patch_ready_files: List[str] = []
            source_diff_nudge_step: Optional[int] = None
            latest_valid_patch_summary = ""
            patch_summary_finish_handoff_enabled = bool(
                getattr(self, "patch_summary_finish_handoff_enabled", False)
            )

            for t in range(max_steps):
                current_step = t + 1
                if self.agent_stop_perf is not None and time.perf_counter() >= self.agent_stop_perf:
                    logger.info(
                        f"[SWEBenchImplementerRunner] wall-clock reserve reached at step "
                        f"{current_step}; stopping ImplementerAgent so the task can grade in time "
                        f"(repo edits preserved for the PlannerAgent's submit)."
                    )
                    break
                logger.log_to_file(LogLevel.INFO, f"Environment Observation:{obs}")

                try:
                    if self.step_timeout:
                        step_result = await asyncio.wait_for(
                            agent.step(
                                observation=obs,
                                history=history,
                                current_step=current_step,
                                max_steps=max_steps,
                            ),
                            timeout=self.step_timeout,
                        )
                    else:
                        step_result = await agent.step(
                            observation=obs,
                            history=history,
                            current_step=current_step,
                            max_steps=max_steps,
                        )
                except asyncio.TimeoutError:
                    logger.error(f"Agent step timed out after {self.step_timeout} seconds")
                    step_record = StepRecord(
                        observation=obs,
                        action={"error": "step_timeout"},
                        reward=0.0,
                        raw_response="step timeout",
                        done=True,
                        info={"error": "step_timeout"},
                        raw_input=None,
                    )
                    history.append(step_record)
                    break

                if isinstance(step_result, (list, tuple)):
                    if len(step_result) == 3:
                        action, raw_response, raw_input = step_result
                    elif len(step_result) == 2:
                        action, raw_response = step_result
                        raw_input = None
                    else:
                        raise ValueError(f"agent.step returned {len(step_result)} values")
                else:
                    raise TypeError(f"agent.step returned unsupported type: {type(step_result)}")

                logger.info(f"\n[SWEBench ImplementerAgent Step {current_step}/{max_steps}] ACTION: {action}")

                sig = _action_signature(action)
                recent_sigs.append(sig)
                if (sig and len(recent_sigs) >= NO_PROGRESS_LIMIT
                        and len(set(recent_sigs[-NO_PROGRESS_LIMIT:])) == 1):
                    logger.info(
                        f"[SWEBenchImplementerAgent] no-progress early stop: identical action "
                        f"repeated {NO_PROGRESS_LIMIT}x through step {current_step}; "
                        f"ending ImplementerAgent (sig={sig[:100]!r})"
                    )
                    history.append(StepRecord(
                        observation=obs,
                        action=action,
                        reward=0.0,
                        raw_response=raw_response,
                        done=True,
                        info={"no_progress_stop": True, "repeated_action": sig[:160]},
                        raw_input=raw_input,
                    ))
                    break

                shared = getattr(agent, "shared_lessons", None)
                thread_id = getattr(agent, "thread_id", 0)
                delegation_id = getattr(agent, "delegation_id", 0)  # 
                patch_summary_channel_enabled = bool(
                    shared is not None
                    and getattr(shared, "feature_flags", {}).get("patch_summary_enabled", False)
                )
                initial_notes = list(action.get("notes") or [])
                if getattr(self, "note_verifier_mode", "deterministic") == "llm":
                    notes_to_emit, verify_status = await verify_notes_llm(
                        initial_notes, getattr(agent, "llm", None)
                    )
                else:
                    notes_to_emit, verify_status = verify_notes(initial_notes)
                notes_verifier_dropped_patch_summary = parse_invalid_evidence_count(verify_status)
                valid_patch_summaries = [
                    n.get("content", "")
                    for n in notes_to_emit
                    if n.get("type") == "PATCH_SUMMARY"
                ]
                notes_emitted_ok = 0
                notes_suppressed_tried = 0
                notes_suppressed_claim = 0
                notes_suppressed_patch_summary = 0
                notes_rejected: List[str] = []
                for note_entry in notes_to_emit:
                    if shared is None:
                        break
                    type_name = note_entry.get("type", "")
                    try:
                        result = await shared.note(
                            thread_id=thread_id,
                            type=type_name,
                            content=note_entry.get("content", ""),
                            delegation_id=delegation_id,
                            task_instruction=(
                                getattr(agent, "current_instruction", "")
                                if getattr(agent, "shared_lessons_render_mode", "") == "patch_selective_unfold"
                                else None
                            ),
                        )
                        if result is None:
                            if type_name == "CLAIM":
                                notes_suppressed_claim += 1
                            elif type_name == "PATCH_SUMMARY":
                                notes_suppressed_patch_summary += 1
                            else:  # TRIED () or any future suppressor
                                notes_suppressed_tried += 1
                        else:
                            notes_emitted_ok += 1
                    except Exception as note_err:
                        notes_rejected.append(f"[{type_name}] {note_err}")

                if patch_summary_channel_enabled and valid_patch_summaries:
                    latest_valid_patch_summary = valid_patch_summaries[-1]

                total_suppressed = (notes_suppressed_tried + notes_suppressed_claim
                                    + notes_suppressed_patch_summary
                                    + notes_verifier_dropped_patch_summary)
                if notes_emitted_ok or notes_rejected or total_suppressed or initial_notes:
                    logger.info(
                        f"[SWEBenchImplementerAgent] notes: {notes_emitted_ok} emitted, "
                        f"{notes_suppressed_tried} tried-suppressed, "
                        f"{notes_suppressed_claim} claim-suppressed, "
                        f"{notes_suppressed_patch_summary} patch-summary-suppressed, "
                        f"{notes_verifier_dropped_patch_summary} patch-summary-invalid-evidence-dropped, "
                        f"{len(notes_rejected)} rejected, verify={verify_status} "
                        f"(initial={len(initial_notes)})"
                    )

                if action.get("action") == "error":
                    err_msg = action.get("params", {}).get("message", "parse error")
                    notes_msg = (
                        f" (NOTE parse: {notes_emitted_ok} ok, {len(notes_rejected)} bad: {notes_rejected[:2]})"
                        if (notes_emitted_ok or notes_rejected) else ""
                    )
                    ack = (
                        f"PARSE_ERROR: {err_msg}{notes_msg}\n"
                        "Your next response MUST contain all three required sections in order: "
                        "DISCUSSION, NOTE, COMMAND."
                    )
                    obs_next = {
                        "message": "parse_error",
                        "output": ack,
                        "state_info": getattr(env, "_aci_manager", None) and env._aci_manager.state.to_status_line() or "(no aci state)",
                        "current_step": current_step,
                        "max_steps": max_steps,
                    }
                    reward = 0.0
                    done = False
                    step_info = {"parse_error": True, "missing_note": True}
                else:
                    try:
                        if self.exec_timeout:
                            obs_next, reward, done, step_info = await asyncio.wait_for(
                                env.step(action), timeout=self.exec_timeout
                            )
                        else:
                            obs_next, reward, done, step_info = await env.step(action)
                    except asyncio.TimeoutError:
                        logger.error(
                            f"[SWEBenchImplementerAgent] env.step timed out after "
                            f"{self.exec_timeout}s at step {current_step}; ending ImplementerAgent."
                        )
                        history.append(StepRecord(
                            observation=obs,
                            action=action,
                            reward=0.0,
                            raw_response=raw_response,
                            done=True,
                            info={"error": "exec_timeout", "timeout_s": self.exec_timeout},
                            raw_input=raw_input,
                        ))
                        break

                if (
                    done
                    and patch_summary_finish_handoff_enabled
                    and patch_summary_channel_enabled
                    and (valid_patch_summaries or latest_valid_patch_summary)
                    and isinstance(step_info, dict)
                    and isinstance(step_info.get("finish_result"), dict)
                ):
                    latest_patch_summary = (
                        valid_patch_summaries[-1]
                        if valid_patch_summaries
                        else latest_valid_patch_summary
                    )
                    finish_result = dict(step_info["finish_result"])
                    if str(finish_result.get("status", "")).lower() == "done":
                        marker = getattr(shared, "mark_patch_summary_final", None)
                        if marker is not None:
                            try:
                                await marker(thread_id, delegation_id, latest_patch_summary)
                            except Exception as mark_err:
                                logger.warning(
                                    "[SWEBenchImplementerRunner] failed to finalize "
                                    f"PATCH_SUMMARY: {mark_err}"
                                )
                    finish_result.setdefault("patch_summary", latest_patch_summary)
                    finish_message = str(finish_result.get("message", "") or "").strip()
                    if latest_patch_summary not in finish_message:
                        finish_result["message"] = (
                            f"{finish_message}\n\nPATCH_SUMMARY: {latest_patch_summary}"
                            if finish_message else f"PATCH_SUMMARY: {latest_patch_summary}"
                        )
                    step_info = dict(step_info)
                    step_info["finish_result"] = finish_result
                    if isinstance(obs_next, dict):
                        obs_next = dict(obs_next)
                        obs_out = str(obs_next.get("output", "") or "")
                        if latest_patch_summary not in obs_out:
                            obs_next["output"] = (
                                f"{obs_out}\nPATCH_SUMMARY: {latest_patch_summary}"
                                if obs_out else f"PATCH_SUMMARY: {latest_patch_summary}"
                            )

                auto_finish_result = None
                patch_ready_note_block = (
                    _patch_ready_block_reason_from_notes(notes_to_emit)
                    if patch_ready_enabled else ""
                )
                if patch_ready_enabled and patch_ready_note_block and patch_ready_step is not None:
                    logger.info(
                        "[SWEBenchImplementerRunner] clearing patch-ready checkpoint: "
                        f"{patch_ready_note_block[:200]}"
                    )
                    patch_ready_step = None
                    patch_ready_summary = ""
                    patch_ready_files = []
                if patch_ready_enabled and valid_patch_summaries and not done and not patch_ready_note_block:
                    source_files = await _current_source_diff_files(env)
                    if source_files:
                        current_summary = valid_patch_summaries[-1]
                        if patch_ready_step is None:
                            patch_ready_step = current_step
                        patch_ready_summary = current_summary
                        patch_ready_files = source_files
                        nudge = (
                            "\n\nPATCH_READY_CHECKPOINT: You emitted a valid "
                            "PATCH_SUMMARY and the working tree has source diffs "
                            f"({', '.join(source_files[:5])}). "
                            "If the last command did not reveal a local FAIL, "
                            "your next command should be `finish done ...` with "
                            "that PATCH_SUMMARY. Do not keep exploring, do not "
                            "run broad suites, and do not chase unrelated local "
                            "failures. Only run one more focused command if the "
                            "last observation specifically left the candidate "
                            "unverified."
                        )
                        if isinstance(obs_next, dict):
                            obs_next = dict(obs_next)
                            obs_next["output"] = str(obs_next.get("output", "")) + nudge
                            obs_next["patch_ready_checkpoint"] = True

                if (
                    patch_ready_enabled
                    and not done
                    and not valid_patch_summaries
                    and not patch_ready_note_block
                ):
                    source_files = await _current_source_diff_files(env)
                    if source_files and (
                        source_diff_nudge_step is None
                        or (current_step - source_diff_nudge_step) >= 3
                    ):
                        source_diff_nudge_step = current_step
                        nudge = (
                            "\n\nSOURCE_DIFF_CHECKPOINT: The working tree now has "
                            f"source diffs ({', '.join(source_files[:5])}) but this "
                            "turn did not emit a valid PATCH_SUMMARY. If this is a "
                            "candidate fix, use the next command for the smallest "
                            "focused verification. If that check passes, emit "
                            "PATCH_SUMMARY with the exact command/result and "
                            "`finish done` immediately. If it fails, emit FAIL and "
                            "continue from that concrete failure."
                        )
                        if isinstance(obs_next, dict):
                            obs_next = dict(obs_next)
                            obs_next["output"] = str(obs_next.get("output", "")) + nudge
                            obs_next["source_diff_checkpoint"] = True

                if (
                    PATCH_READY_AUTO_FINISH_AFTER is not None
                    and patch_ready_step is not None
                    and patch_ready_enabled
                    and not done
                ):
                    if (current_step - patch_ready_step) >= PATCH_READY_AUTO_FINISH_AFTER:
                        live_source_files = await _current_source_diff_files(env)
                        if not live_source_files:
                            logger.info(
                                "[SWEBenchImplementerRunner] clearing patch-ready checkpoint: "
                                "source diffs disappeared before auto-finish"
                            )
                            patch_ready_step = None
                            patch_ready_summary = ""
                            patch_ready_files = []
                        else:
                            patch_ready_files = live_source_files
                            if patch_ready_note_block:
                                logger.info(
                                    "[SWEBenchImplementerRunner] patch-ready auto-finish blocked: "
                                    f"{patch_ready_note_block[:200]}"
                                )
                            else:
                                msg = (
                                    "Patch-ready checkpoint persisted for "
                                    f"{current_step - patch_ready_step} steps after a valid "
                                    "PATCH_SUMMARY with source diffs. Returning candidate to "
                                    "PlannerAgent instead of continuing exploration. "
                                    f"Files: {', '.join(patch_ready_files[:8])}. "
                                    f"Summary: {patch_ready_summary}"
                                )
                                auto_finish_result = {
                                    "status": "partial",
                                    "patch_ready_auto_finish": True,
                                    "source_files": patch_ready_files[:8],
                                    "patch_summary": patch_ready_summary,
                                    "message": msg,
                                    "completed": [],
                                    "issues": [
                                        "patch_ready_candidate: runner synthesized a "
                                        "partial candidate handoff after grace-period "
                                        "verification; ImplementerAgent did not explicitly call finish."
                                    ],
                                }

                step_record = StepRecord(
                    observation=obs,
                    action=action,
                    reward=reward,
                    raw_response=raw_response,
                    done=done,
                    info=step_info,
                    raw_input=raw_input,
                )
                history.append(step_record)
                total_reward += reward
                obs = obs_next

                if auto_finish_result is not None:
                    history.append(StepRecord(
                        observation={
                            "message": "patch_ready_auto_finish",
                            "output": auto_finish_result["message"],
                            "current_step": len(history) + 1,
                            "max_steps": max_steps,
                        },
                        action={"action": "patch_ready_auto_finish", "params": {}},
                        reward=0.0,
                        raw_response="",
                        done=True,
                        info={
                            "finished": True,
                            "patch_ready_auto_finish": True,
                            "finish_result": auto_finish_result,
                        },
                        raw_input=None,
                    ))
                    logger.info(
                        "[SWEBenchImplementerRunner] patch-ready auto-finish after "
                        f"step {current_step}; files={patch_ready_files[:8]}"
                    )
                    break

                if done:
                    break

            if history and not history[-1].done:
                last = history[-1]
                last_sig = _action_signature(last.action)
                last_obs = obs
                if isinstance(last_obs, dict):
                    obs_tail = str(last_obs.get("output", "") or last_obs.get("message", ""))[-400:]
                else:
                    obs_tail = ("" if last_obs is None else str(last_obs))[-400:]
                msg = (
                    f"Reached max_steps ({max_steps}) without an explicit finish. "
                    f"Last action: {last_sig[:160]}. Last observation tail: {obs_tail}"
                )
                if patch_summary_channel_enabled and latest_valid_patch_summary:
                    if patch_summary_finish_handoff_enabled:
                        msg += f" Latest PATCH_SUMMARY: {latest_valid_patch_summary}"
                finish_result = {
                    "status": "partial",
                    "message": msg,
                    "completed": [],
                    "issues": [
                        "max_steps_exhausted: ImplementerAgent did not call finish; current "
                        "repo edits are preserved in the container for the next attempt."
                    ],
                }
                if (
                    patch_summary_finish_handoff_enabled
                    and patch_summary_channel_enabled
                    and latest_valid_patch_summary
                ):
                    finish_result["patch_summary"] = latest_valid_patch_summary
                history.append(StepRecord(
                    observation={"message": "max_steps_exhausted", "output": msg,
                                 "current_step": len(history) + 1, "max_steps": max_steps},
                    action={"action": "max_steps_exhausted", "params": {}},
                    reward=0.0,
                    raw_response="",
                    done=True,
                    info={"finished": True, "max_steps_exhausted": True,
                          "finish_result": finish_result},
                    raw_input=None,
                ))
                logger.info(
                    f"[SWEBenchImplementerRunner] max_steps ({max_steps}) exhausted without "
                    f"finish; appended deterministic partial handoff."
                )

            usage_summary = agent.llm.get_usage_summary()
            cost = usage_summary.get("total_cost", 0.0)
            input_t = usage_summary.get("total_input_tokens", 0)
            output_t = usage_summary.get("total_output_tokens", 0)
            mem = getattr(agent, "memory", None)
            sum_llm = getattr(mem, "summarizer_llm", None) if mem is not None else None
            if sum_llm is not None and sum_llm is not agent.llm:
                try:
                    s_usage = sum_llm.get_usage_summary()
                    cost += float(s_usage.get("total_cost", 0.0) or 0.0)
                    input_t += int(s_usage.get("total_input_tokens", 0) or 0)
                    output_t += int(s_usage.get("total_output_tokens", 0) or 0)
                except Exception as e:
                    logger.warning(
                        f"[SWEBenchImplementerRunner] summarizer usage_summary failed: {e}"
                    )
            result = LevelResult(
                model=usage_summary.get("model", ""),
                total_reward=total_reward,
                steps=len(history),
                done=history[-1].done if history else False,
                trace=history,
                cost=cost,
                input_tokens=input_t,
                output_tokens=output_t,
            )
            logger.info(f"[SWEBenchImplementerRunner] ImplementerAgent completed: steps={result.steps}, reward={result.total_reward}")
            return result

        except Exception as e:
            logger.error(f"[SWEBenchImplementerRunner] ImplementerAgent execution failed: {e}", exc_info=True)
            raise
