"""Provider adapters only. Worker count never changes authentication or runtime."""

import json
from pathlib import Path
import subprocess

from runtime import codex_api
from runtime.config import check_authentication
from runtime import claude_api as claude
from runtime.containers import Container, collect_native_record
from runtime.native_network import collect_transport
from benchmark.terminal_task import image_name

ROOT = Path(__file__).resolve().parents[1]


def validate_key_file(key):
    key = key.resolve()
    if not key.is_file() or key.stat().st_size == 0:
        raise ValueError("Supply a nonempty credential file")
    if key.stat().st_mode & 0o077:
        raise ValueError("Credential file must be private (chmod 600)")
    if key.is_relative_to(ROOT) and not key.is_relative_to(ROOT / ".local"):
        raise ValueError("Keep credentials outside the checkout or in ignored .local")


def runtime(harness):
    if harness == "codex":
        return codex_api
    if harness == "claude-code":
        return claude
    raise ValueError("Unknown harness")


def task_image(task, harness):
    return image_name(task) if harness == "codex" else f"delm-{task.name}:claude-{claude.VERSION}"


def validate_overlay(image, base, harness):
    adapter = runtime(harness)
    labels = image["Config"].get("Labels") or {}
    if (labels.get("delm.harness"), labels.get("delm.runtime.version"),
            labels.get("delm.driver"), labels.get("delm.base-task-image")) != (
            harness, adapter.VERSION, adapter.DRIVER_REVISION, base["Id"]):
        raise ValueError("Image does not match the selected runtime and base task image")
    if (image["Config"].get("WorkingDir") != base["Config"].get("WorkingDir")
            or image.get("Architecture") != base.get("Architecture")):
        raise ValueError("Runtime image changed the task working directory or architecture")


def worker_mounts(harness, key, authentication="api-key"):
    check_authentication(harness, authentication)
    adapter = runtime(harness)
    credential_path = adapter.KEY_PATH if authentication == "api-key" else adapter.AUTH_PATH
    worker = ROOT / "runtime" / ("codex_api.py" if harness == "codex" else "claude_api.py")
    return [(ROOT / "runtime/__init__.py", "/tools/runtime/__init__.py"),
            (ROOT / "runtime/config.py", "/tools/runtime/config.py"),
            (worker, "/tools/worker.py"), (ROOT / "runtime/native_worker.py", "/tools/native_worker.py"),
            (ROOT / "board/context_client.py", "/tools/context_client.py"),
            (ROOT / "board/context.sh", "/tools/context.sh"), (key, credential_path)]


class WorkerContainer(Container):
    def preflight(self, storage_mb, context_url):
        host = "api.openai.com" if self.harness == "codex" else "api.anthropic.com"
        if self.authentication == "chatgpt":
            host = "chatgpt.com"
        code = (
            "import json,os,shutil,socket,ssl,sys; from pathlib import Path; "
            "sys.path.insert(0,'/tools'); import worker; worker.clean_environment(os.environ); "
            "assert not Path('/tests').exists() and not Path('/solution').exists(); "
            f"assert shutil.disk_usage({self.workdir!r}).free >= {storage_mb}*1024*1024; "
            "runtime=worker.preflight(os.environ); "
            "tls=ssl.create_default_context(cafile='/opt/native/share/ca-certificates.crt'); "
            f"s=tls.wrap_socket(socket.create_connection(({host!r},443),timeout=5),server_hostname={host!r}); s.close(); "
            "import urllib.request; "
            f"assert json.load(urllib.request.urlopen({context_url + '/health'!r},timeout=5))['ready']; "
            "print(json.dumps(runtime))"
        )
        result = subprocess.run(["docker", "exec", "-e", "HOME=/root", "-e", "LD_LIBRARY_PATH=/opt/native/lib",
                                 "-e", "PATH=/opt/agent-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                                 self.name, "/opt/native/bin/python3.12", "-c", code],
                                capture_output=True, text=True, timeout=40)
        (self.folder / "preflight.json").write_text(json.dumps(dict(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)))
        if result.returncode:
            raise RuntimeError(f"{self.harness} preflight failed for {self.folder.name}; no model started")
        (self.folder / "runtime.json").write_text(json.dumps(json.loads(result.stdout), indent=2))

    def start(self, image_env_path):
        log = (self.folder / "process.log").open("x")
        try:
            process = subprocess.Popen([
                "docker", "exec", "-e", "HOME=/root", "-e", "LD_LIBRARY_PATH=/opt/native/lib",
                "-e", f"PATH=/opt/agent-bin:{image_env_path}", self.name,
                "/opt/native/bin/python3.12", "/tools/worker.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        except BaseException:
            log.close()
            raise
        return process, log


def collect_record(harness, container, row, model):
    if harness == "codex":
        if row.get("authentication") != container.authentication:
            raise ValueError("Effective Codex authentication differs from the requested method")
        usage = collect_native_record(container.name, container.folder)
        session = codex_api.validate_record(container.folder / "native-trajectory.jsonl", model)
        usage = dict(usage, session_id=session)
        row["usage_agrees"] = sum(row.get("tokens", {}).values()) == usage["total_tokens"]
        # Diagnostic-log failure must not discard the independently captured usage.
        try:
            row["model_transport_failures"] = collect_transport(container.name, container.folder)
        except Exception as error:
            row["transport_record_error"] = f"{type(error).__name__}: {error}"
    else:
        if row.get("authentication", "api-key") != container.authentication:
            raise ValueError("Effective Claude authentication differs from the requested method")
        usage = claude.collect_record(container.folder, container.authentication, model)
        row["usage_agrees"] = usage == row.get("native_usage") and sum(row.get("tokens", {}).values()) == usage["total_tokens"]
    row["native_usage"] = usage


def validate_records(run):
    if run.get("runner_timeout"):
        raise ValueError("Runner deadline interrupted execution")
    rows = run["workers"]
    sessions = [r.get("native_usage", {}).get("session_id") for r in rows]
    if not all(sessions) or len(set(sessions)) != run["count"]:
        raise ValueError("Expected a distinct native session for every worker")
    for row in rows:
        if any(row.get(k) for k in ("error", "timed_out", "shutdown_error", "record_error", "result_record_error",
                                    "grading_error", "session_state_error", "transport_record_error")):
            raise ValueError("Worker contains an execution or recording error")
        if row.get("grading", {}).get("workspace_copy_error"):
            raise ValueError("Submission capture was incomplete")
        if run["harness"] == "claude-code":
            import math
            usage = row["native_usage"]
            if usage.get("partial") is not False or not math.isfinite(usage["cost_usd"]) or usage["cost_usd"] < 0:
                raise ValueError("Missing complete native usage or cost")
