"""One unmodified Codex session inside a task container, driven through Humanize.

Runs the prompt in /records/prompt.txt to completion, records every native
event, and writes the result. Nothing is ever added to the session after the
prompt; the shared board is reached only by the agent's own shell commands.
"""

import dataclasses
import json
import os
import signal
import time
from pathlib import Path

from runtime.config import check_authentication


class AgentDeadline(Exception):
    pass


def model_settings(environment):
    model = environment["AGENT_MODEL"]
    effort = environment["AGENT_EFFORT"]
    if not model or effort not in ("low", "medium", "high", "xhigh"):
        raise ValueError("An explicit model and supported reasoning effort are required")
    return dict(model=model, effort=effort)


def prepare_task_skills(environment, home=Path('/root')):
    """Expose only the skill directory declared by the official task."""
    declared = environment.get('TASK_SKILLS_DIR')
    if not declared:
        return None
    source = Path(declared)
    if not source.is_absolute() or not source.is_dir():
        raise RuntimeError('Task skills directory must be an existing absolute directory')
    source = source.resolve()
    target = home / '.agents/skills'
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source:
            raise RuntimeError('Unexpected existing user skills directory')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)
    return source


def check_skills(inventory, workspace, task_skills=None):
    entries = inventory.get("data", [])
    if len(entries) != 1 or entries[0].get("cwd") != workspace:
        raise RuntimeError("Missing skill inventory for the task workspace")
    if entries[0].get("errors"):
        raise RuntimeError("Codex could not inspect all skill locations")
    found_task = set()
    for skill in entries[0]["skills"]:
        path = Path(skill['path']).resolve()
        bundled = skill['scope'] == 'system' and Path('/root/.codex/skills/.system') in path.parents
        declared = (task_skills is not None and skill['scope'] in ('user', 'repo')
                    and task_skills in path.parents)
        if not bundled and not declared:
            raise RuntimeError('Unexpected skill outside bundled or task-declared directories; no model turn started')
        if declared:
            found_task.add(path)
    if task_skills is not None:
        expected = {p.resolve() for p in task_skills.rglob('SKILL.md')}
        if not expected or expected != found_task:
            raise RuntimeError('Task-declared skills were not fully loaded; no model turn started')


def notice(event, elapsed_seconds, sequence):
    return dict(event=event, elapsed_seconds=elapsed_seconds, sequence=sequence)


def require_account(server, authentication="api-key"):
    """Check effective auth before the first model turn; persist no account data."""
    expected = {"api-key": "apiKey", "chatgpt": "chatgpt"}[check_authentication("codex", authentication)]
    response = server.call("account/read", {"refreshToken": False})
    if (response.get("account") or {}).get("type") != expected or response.get("requiresOpenaiAuth") is not True:
        raise RuntimeError(f"Codex must use the requested {authentication} account")
    return authentication


def main():
    from hmz.agents import CodexAgent, CodexAgentConfig

    os.environ.pop("LD_LIBRARY_PATH", None)  # start the relocated Python only; task children unaffected
    task_skills = prepare_task_skills(os.environ)
    config = CodexAgentConfig(**model_settings(os.environ), web_search=False)
    agent = CodexAgent(config, name=os.environ["WORKER_NAME"])
    prompt = Path("/records/prompt.txt").read_text()
    started = time.monotonic_ns()
    result = dict(started_ns=started, model=config.model, effort=config.effort)
    try:
        result["authentication"] = require_account(agent.server, os.environ.get("AGENT_AUTHENTICATION", "api-key"))
        workspace = os.environ["TASK_WORKSPACE"]
        inventory = agent.server.call("skills/list", {"cwds": [workspace], "forceReload": True})
        Path("/records/skills.json").write_text(json.dumps(inventory, indent=2))
        check_skills(inventory, workspace, task_skills)
        session = agent.new(cwd=workspace)
        result["model_turn_start_seconds"] = (time.monotonic_ns() - started) / 1e9
        print(json.dumps(notice("model_start", result["model_turn_start_seconds"], 0)), flush=True)
        def expired(signum, frame):
            raise AgentDeadline("Agent task time limit reached")
        signal.signal(signal.SIGALRM, expired)
        remaining = float(os.environ["AGENT_TIMEOUT_SECONDS"])
        signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining))
        with Path("/records/trajectory.jsonl").open("x") as output:
            for sequence, event in enumerate(session.stream(prompt), 1):
                elapsed = (time.monotonic_ns() - started) / 1e9
                output.write(json.dumps(dict(sequence=sequence, elapsed_seconds=elapsed,
                                             event=dataclasses.asdict(event)), default=str) + "\n")
                output.flush()
                if event.kind == "result":
                    result.update(agent_finished_seconds=elapsed, tokens=dict(event.tokens),
                                  answer=event.text, completed=True)
                    print(json.dumps(notice("result", elapsed, sequence)), flush=True)
    except AgentDeadline as error:
        result.update(timed_out=True, error=str(error),
                      agent_finished_seconds=(time.monotonic_ns() - started) / 1e9)
        print(json.dumps(notice("deadline", result["agent_finished_seconds"], 0)), flush=True)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        try:
            agent.stop()
        except Exception as error:
            result["shutdown_error"] = f"{type(error).__name__}: {error}"
        result["process_elapsed_seconds"] = (time.monotonic_ns() - started) / 1e9
        result["elapsed_seconds"] = result.get("agent_finished_seconds", result["process_elapsed_seconds"])
        Path("/records/result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
