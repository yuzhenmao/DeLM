"""Pinned Claude worker with explicit API-key or subscription authentication."""

import dataclasses
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from collections.abc import Mapping
from decimal import Decimal

from runtime.config import CLI_VERSIONS, DEFAULT_MODELS, DRIVER_REVISION, EFFORT, check_authentication, check_driver

VERSION = CLI_VERSIONS["claude-code"]
MODEL = DEFAULT_MODELS["claude-code"]
KEY_PATH = "/run/secrets/anthropic_api_key"
AUTH_PATH = "/run/secrets/claude_oauth_token"
DENIED_TOOLS = {"Agent", "ScheduleWakeup", "CronCreate", "CronDelete", "CronList", "WebSearch", "WebFetch"}


# Same as runner.write_json; this file runs alone inside the worker container.
def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def command(upstream, authentication="api-key", model=MODEL):
    """Use the pinned driver's protocol, with explicit experiment settings."""
    argv = list(upstream)
    for flag, expected in (("--model", model), ("--effort", EFFORT)):
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != expected:
            raise ValueError(f"Unexpected Claude command: {flag}")
    if argv.count("--settings") != 1 or "--fallback-model" in argv:
        raise ValueError("Unexpected Claude settings or fallback model")
    settings = json.loads(argv[argv.index("--settings") + 1])
    if settings != {"fastMode": False}:
        raise ValueError("Expected the pinned driver's default-speed settings")
    if argv.count("--disallowedTools") != 1:
        raise ValueError("Missing tool restrictions")
    if set(argv[argv.index("--disallowedTools") + 1].split(",")) != DENIED_TOOLS:
        raise ValueError("Unexpected tool restrictions")
    settings.update(availableModels=[model])
    if check_authentication("claude-code", authentication) == "api-key":
        settings["apiKeyHelper"] = f"/bin/cat {KEY_PATH}"
    argv[argv.index("--settings") + 1] = json.dumps(settings)
    # No operator settings, plugins, or MCP connections are inherited. The task's
    # declared skills are exposed explicitly during preflight, not copied from a login.
    argv += ["--setting-sources", "user", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
    return argv


def clean_environment(environment):
    """Prevent inherited provider variables from choosing authentication or effort."""
    for key in list(environment):
        if key.startswith(("ANTHROPIC_", "CLAUDE_")):
            del environment[key]
    environment.pop("LD_LIBRARY_PATH", None)
    environment["HOME"] = "/root"
    environment["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    environment["DISABLE_AUTOUPDATER"] = "1"
    environment["CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL"] = "1"


def prepare_auth(environment):
    """Load only the selected mounted credential; never put it in argv or records."""
    authentication = check_authentication("claude-code", environment.get("AGENT_AUTHENTICATION", "api-key"))
    path = Path(AUTH_PATH if authentication == "claude-oauth" else KEY_PATH)
    credential = path.read_text().strip()
    if not credential or any(c.isspace() for c in credential):
        raise ValueError("Supply a raw single-line Claude credential")
    if authentication == "claude-oauth":
        if not credential.startswith("sk-ant-oat01-") or len(credential) <= len("sk-ant-oat01-"):
            raise ValueError("Supply the raw OAuth token from claude setup-token, not an API key or login JSON")
        environment["CLAUDE_CODE_OAUTH_TOKEN"] = credential
    elif credential.startswith(("{", "[", "sk-ant-oat01-")):
        raise ValueError("Supply a raw API key, not subscription credentials")
    return authentication


def require_subscription(environment):
    """Check the pinned CLI's credential source without recording account details."""
    result = subprocess.run(["claude", "auth", "status", "--json"], env=dict(environment),
                            capture_output=True, text=True, timeout=20)
    try:
        status = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise ValueError("Claude did not report a valid authentication status") from None
    if (result.returncode or not isinstance(status, dict) or status.get("loggedIn") is not True
            or status.get("authMethod") != "oauth_token" or status.get("apiProvider") != "firstParty"):
        raise ValueError("Claude must use the requested subscription OAuth token")
    return dict(auth_method="oauth_token", api_provider="firstParty")


def prepare_skills(environment, home):
    declared = environment.get("TASK_SKILLS_DIR")
    target = home / ".claude/skills"
    if target.exists() or target.is_symlink():
        if not declared or not target.is_symlink() or target.resolve() != Path(declared).resolve():
            raise ValueError("Unexpected Claude user skills")
    if not declared:
        return []
    source = Path(declared)
    if not source.is_absolute() or not source.is_dir():
        raise ValueError("Missing task-declared skills directory")
    skills = sorted(source.rglob("SKILL.md"))
    if not skills:
        raise ValueError("Task-declared skills directory contains no skills")
    if not target.is_symlink():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve(), target_is_directory=True)
    return [str(p.resolve()) for p in skills]


def skill_names(paths):
    # The pinned CLI's startup inventory identifies skills by directory name,
    # including when YAML frontmatter supplies a different display name.
    names = [Path(path).parent.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("Task skill directory names must be distinct")
    return sorted(names)


def preflight(environment, home=Path("/root")):
    """Local runtime validation only: no model request and no secret in records."""
    model = environment.get("AGENT_MODEL")
    if not isinstance(model, str) or not model.strip() or environment.get("AGENT_EFFORT") != EFFORT:
        raise ValueError("Claude requires an explicit model and xhigh effort")
    for path in (home / ".claude/.credentials.json", home / ".claude.json", home / ".claude/settings.json",
                 home / ".claude/settings.local.json", home / ".claude/plugins", home / ".claude/agents",
                 home / ".claude/commands"):
        if path.exists():
            raise ValueError("Claude home must not contain a login or inherited settings")
    authentication = prepare_auth(environment)
    version = subprocess.check_output(["claude", "--version"], text=True, timeout=20).strip()
    if version.split()[0] != VERSION:
        raise ValueError(f"Expected Claude Code {VERSION}, found {version}")
    check_driver()
    skills = prepare_skills(environment, home)
    account = require_subscription(environment) if authentication == "claude-oauth" else {}
    return dict(claude_version=VERSION, driver_revision=DRIVER_REVISION,
                authentication="claude-oauth" if authentication == "claude-oauth" else "apiKeyHelper",
                subscription_credentials=authentication == "claude-oauth", **account,
                model=model, effort=EFFORT, fast_mode=False, task_skills=skills,
                task_skill_names=skill_names(skills))


class StreamAudit:
    """Validate native session identity and final cumulative usage exactly once."""

    def __init__(self, expected_skills=(), authentication="api-key", model=MODEL):
        self.authentication = check_authentication("claude-code", authentication)
        self.model = model
        self.initial = None
        self.final = None
        self.models = set()
        self.expected_skills = set(expected_skills)

    def observe(self, event):
        if not isinstance(event, dict):
            raise ValueError("Claude emitted a non-object event")
        if event.get("type") == "system" and event.get("subtype") == "init":
            if self.initial is not None:
                raise ValueError("Unexpected additional Claude session")
            self.initial = event
            # OAuth has no API key: the pinned CLI reports "none" here.
            # Subscription identity is checked separately with `auth status`.
            if (event.get("model"), event.get("claude_code_version"), event.get("apiKeySource"),
                    event.get("fast_mode_state")) != (self.model, VERSION,
                    "none" if self.authentication == "claude-oauth" else "apiKeyHelper", "off"):
                raise ValueError("Effective model, version, authentication, or fast mode differs")
            if not event.get("session_id") or event.get("mcp_servers"):
                raise ValueError("Missing session identity or unexpected MCP server")
            if not self.expected_skills.issubset(set(event.get("skills", []))):
                raise ValueError("Task-declared skills were not loaded by Claude")
        elif event.get("type") == "assistant":
            message = event.get("message", {})
            model = message.get("model")
            # Claude uses <synthetic> for local status messages, not API responses.
            if model and model != "<synthetic>":
                self.models.add(model)
                if model != self.model:
                    raise ValueError("Primary assistant changed model")
            for item in message.get("content", []):
                if item.get("type") == "tool_use" and item.get("name") in DENIED_TOOLS | {"Task"}:
                    raise ValueError("Claude attempted a disabled tool")
        elif event.get("type") == "result":
            if self.final is not None:
                raise ValueError("Unexpected additional final result")
            self.final = event
            if event.get("is_error") is not False or event.get("subtype") != "success":
                raise ValueError("Claude session did not complete successfully")
            if event.get("stop_reason") not in (None, "end_turn", "stop_sequence"):
                raise ValueError("Claude session stopped before completion")

    def usage(self):
        if not self.initial or not self.final or self.models != {self.model}:
            raise ValueError("Missing native session, assistant, or final result")
        if self.final.get("session_id") != self.initial["session_id"]:
            raise ValueError("Final result belongs to another session")
        models = self.final.get("modelUsage")
        if not isinstance(models, dict) or self.model not in models:
            raise ValueError("Missing native modelUsage totals")
        fields = ("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "outputTokens")
        totals = dict.fromkeys(fields, 0)
        costs = []
        for usage in models.values():
            for field in fields:
                value = usage.get(field)
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid native usage: {field}")
                totals[field] += value
            cost = Decimal(str(usage.get("costUSD")))
            if not cost.is_finite() or cost < 0:
                raise ValueError("Invalid native model cost")
            costs.append(cost)
        cost = Decimal(str(self.final.get("total_cost_usd")))
        if not cost.is_finite() or cost < 0 or abs(cost - sum(costs)) > Decimal("0.000001"):
            raise ValueError("Model costs disagree with the final session cost")
        input_tokens = sum(totals[f] for f in fields[:3])
        return dict(input_tokens=input_tokens, cached_input_tokens=totals["cacheReadInputTokens"],
                    cache_creation_input_tokens=totals["cacheCreationInputTokens"],
                    uncached_input_tokens=input_tokens - totals["cacheReadInputTokens"],
                    output_tokens=totals["outputTokens"], total_tokens=input_tokens + totals["outputTokens"],
                    cost_usd=float(cost), model_usage=models, partial=False,
                    session_id=self.initial["session_id"])


def collect_record(folder, authentication=None, model=None):
    runtime = json.loads((folder / "runtime.json").read_text())
    recorded_model = runtime.get("model", MODEL)
    if model is not None and model != recorded_model:
        raise ValueError("Recorded Claude model differs from the requested model")
    recorded = "claude-oauth" if runtime.get("authentication") == "claude-oauth" else "api-key"
    if authentication is not None and authentication != recorded:
        raise ValueError("Recorded Claude authentication differs from the requested method")
    if recorded == "claude-oauth" and (runtime.get("auth_method") != "oauth_token"
            or runtime.get("api_provider") != "firstParty" or runtime.get("subscription_credentials") is not True):
        raise ValueError("Missing validated Claude subscription authentication")
    audit = StreamAudit(runtime.get("task_skill_names", []), recorded, recorded_model)
    with (folder / "native-trajectory.jsonl").open() as stream:
        for line in stream:
            audit.observe(json.loads(line))
    return audit.usage()


def session_type(records, audit):
    from hmz.agents.claude import ClaudeCodeSession

    class RecordedSession(ClaudeCodeSession):
        def _command(self):
            upstream = super()._command()
            effective = command(upstream, audit.authentication, audit.model)
            write_json(records / "claude-command.json", dict(upstream=upstream, effective=effective))
            return effective

        def _read(self, line):
            with (records / "native-trajectory.jsonl").open("a") as stream:
                stream.write(line.rstrip("\n") + "\n")
            event = json.loads(line)
            audit.observe(event)
            yield from super()._read(line)

    return RecordedSession


class AgentDeadline(Exception):
    pass


def main(records=Path("/records")):
    from hmz.agents import ClaudeCodeAgent, ClaudeCodeAgentConfig

    clean_environment(os.environ)
    started = time.monotonic_ns()
    elapsed = lambda: (time.monotonic_ns() - started) / 1e9
    model = os.environ.get("AGENT_MODEL", MODEL)
    result = dict(started_ns=started, model=model, effort=EFFORT, harness="claude-code", completed=False)
    agent = None
    audit = StreamAudit()

    def expired(signum, frame):
        raise AgentDeadline("Official agent time limit reached")

    signal.signal(signal.SIGALRM, expired)
    try:
        # Runtime checks were performed for every container before launch.
        runtime = json.loads((records / "runtime.json").read_text())
        if runtime.get("claude_version") != VERSION or runtime.get("driver_revision") != DRIVER_REVISION:
            raise ValueError("Missing validated runtime")
        if not model.strip() or runtime.get("model", MODEL) != model:
            raise ValueError("Validated Claude model differs from the requested model")
        authentication = check_authentication("claude-code", os.environ.get("AGENT_AUTHENTICATION", "api-key"))
        expected = "claude-oauth" if authentication == "claude-oauth" else "apiKeyHelper"
        if runtime.get("authentication", "apiKeyHelper") != expected:
            raise ValueError("Validated Claude authentication differs from the requested method")
        if authentication == "claude-oauth":
            prepare_auth(os.environ)
            require_subscription(os.environ)
        result["authentication"] = authentication
        audit.authentication = authentication
        audit.model = model
        audit.expected_skills = set(runtime.get("task_skill_names", []))
        records.joinpath("native-trajectory.jsonl").touch(exist_ok=False)
        session_class = session_type(records, audit)

        agent = ClaudeCodeAgent(ClaudeCodeAgentConfig(model=model, effort=EFFORT,
                              service_tier="default", web_search=False, goals=False),
                              name=os.environ["WORKER_NAME"])
        session = session_class(agent, cwd=os.environ["TASK_WORKSPACE"])
        timeout = float(os.environ["AGENT_TIMEOUT_SECONDS"])
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid official agent time limit")
        result["model_turn_start_seconds"] = elapsed()
        signal.setitimer(signal.ITIMER_REAL, timeout)
        print(json.dumps(dict(event="model_start", elapsed_seconds=elapsed())), flush=True)
        with (records / "trajectory.jsonl").open("x") as stream:
            for sequence, event in enumerate(session.stream((records / "prompt.txt").read_text()), 1):
                record = {f.name: dict(v) if isinstance(v := getattr(event, f.name), Mapping) else v
                          for f in dataclasses.fields(event)}
                stream.write(json.dumps(dict(sequence=sequence, elapsed_seconds=elapsed(), event=record), default=str) + "\n")
                stream.flush()
                if event.kind == "result":
                    usage = audit.usage()
                    if sum(event.tokens.values()) != usage["total_tokens"]:
                        raise ValueError("Driver usage disagrees with native session usage")
                    result.update(agent_finished_seconds=elapsed(), tokens=dict(event.tokens),
                                  native_usage=usage, answer=event.text, completed=True)
                    print(json.dumps(dict(event="result", elapsed_seconds=elapsed(), sequence=sequence)), flush=True)
        if not result["completed"]:
            raise ValueError("Session ended without a complete final response")
    except AgentDeadline as error:
        result.update(timed_out=True, completed=False, error=str(error), agent_finished_seconds=elapsed())
        print(json.dumps(dict(event="deadline", elapsed_seconds=elapsed())), flush=True)
    except Exception as error:
        result.update(completed=False, error=f"{type(error).__name__}: {error}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if agent:
            try:
                agent.stop()
            except Exception as error:
                result.update(completed=False, shutdown_error=f"{type(error).__name__}: {error}")
        result["process_elapsed_seconds"] = elapsed()
        result["elapsed_seconds"] = result.get("agent_finished_seconds", elapsed())
        write_json(records / "result.json", result)
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
