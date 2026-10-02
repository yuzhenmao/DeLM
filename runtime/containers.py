"""Task containers, original-file capture, and board HTTP/lifecycle helpers."""

import io
import json
import socket
import subprocess
import tarfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler

from board.shared_context import BoardError, OriginalReadError


def handler(context, actors):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.reply(200, {"ready": True} if self.path == "/health" else {"error": "not found"})

        def do_POST(self):
            actor = actors.get(self.path.removeprefix("/"))
            if actor is None:
                self.reply(403, {"error": "unknown identity"})
                return
            try:
                p = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                action = p["action"]
                if action == "read":
                    answer = context.read(actor, full=p.get("all", False), wait_seconds=p.get("wait") or 0,
                                          target=p.get("for") or None)
                elif action == "status":
                    answer = context.status(actor, p["text"])
                elif action == "claim":
                    answer = context.claim(actor, p["item"])
                elif action == "reserve":
                    answer = context.reserve(actor, p["subject"], p["detail"])
                elif action == "done":
                    answer = context.done(actor, p["item"], p["summary"], p.get("files"),
                                          request_id=p.get("request_id"))
                elif action == "note":
                    answer = context.note(actor, p["type"], p["text"], p["scope"], p["basis"], p.get("files"),
                                          request_id=p.get("request_id"))
                elif action == "attached":
                    answer = context.attached(actor, p["id"])
                elif action == "expand":
                    answer = context.expand(actor, p["id"], p.get("paths"))
                    answer["view"] = context.view_after(actor, "expand")
                elif action == "apply":
                    answer = context.apply(actor, p["id"], p["local"], p.get("paths"))
                    if not answer["applied"]:
                        answer["view"] = context.view_after(actor, "apply")
                elif action == "apply-error":
                    context.apply_error(actor, p["id"], p["error"]); answer = {"saved": True}
                elif action == "apply-result":
                    answer = context.apply_result(actor, p["id"], p["written"], p["success"])
                elif action == "publish":
                    answer = context.publish(actor, p["items"])
                else:
                    raise BoardError("unknown action")
                self.reply(200, answer)
            except (BoardError, KeyError, ValueError) as error:
                self.reply(400, {"error": str(error)})

        def reply(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler


def collect_native_record(container, worker):
    archive_bytes = subprocess.check_output(["docker", "cp", f"{container}:/root/.codex/sessions", "-"],
                                            stderr=subprocess.PIPE, timeout=30)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        records = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".jsonl")]
        if len(records) != 1:
            raise RuntimeError(f"Expected one native session record, found {len(records)}")
        raw = archive.extractfile(records[0]).read()
    (worker / "native-trajectory.jsonl").write_bytes(raw)
    usage = None
    for line in raw.splitlines():
        record = json.loads(line)
        payload = record.get("payload", {})
        if record.get("type") == "event_msg" and payload.get("type") == "token_count" and payload.get("info"):
            usage = payload["info"]["total_token_usage"]
    if usage is None:
        raise RuntimeError("Native session has no usage measurement")
    return usage


def board_address():
    """Where the board listens: the Docker bridge gateway on Docker Engine (Linux), the loopback on Docker Desktop.

    Containers reach the host as `host.docker.internal`, which Docker Desktop
    defines by itself and Docker Engine defines only when a container is
    started with `--add-host host.docker.internal:host-gateway`; the runner
    always passes it. On Docker Engine that name resolves to the bridge
    gateway, so the board must listen there.
    """
    try:
        gateway = subprocess.check_output(["docker", "network", "inspect", "bridge", "--format",
                                           "{{(index .IPAM.Config 0).Gateway}}"], text=True, timeout=15).strip()
        with socket.socket() as probe:
            probe.bind((gateway, 0))
        return gateway
    except (subprocess.SubprocessError, OSError, ValueError):
        return "127.0.0.1"


class OriginalFiles:
    """The task's original files, read from a container that is created and never started.

    The board fetches a file from here when an agent registers it, so that a
    read can say whether it is new or modified and expand can show a diff.
    Nothing is copied in bulk.
    """

    def __init__(self, image, workdir):
        self.name = "delm-original-" + uuid.uuid4().hex[:12]
        self.workdir = workdir.rstrip("/")
        self.cache = {}
        self.cache_lock = threading.Lock()
        subprocess.run(["docker", "create", "--name", self.name, "--label", "delm.workspace.research=true", image, "true"],
                       check=True, capture_output=True, text=True, timeout=60)

    def fetch(self, path):
        """Cache immutable originals and confirmed absence, but never read failures."""
        with self.cache_lock:
            if path not in self.cache:
                self.cache[path] = self._fetch(path)
            return self.cache[path]

    def _fetch(self, path):
        """Return original bytes or confirmed absence, and distinguish capture failures."""
        target = path if path.startswith("/") else f"{self.workdir}/{path}"
        try:
            result = subprocess.run(["docker", "cp", f"{self.name}:{target}", "-"], capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OriginalReadError(f"Could not read {target}: {error}") from error
        if result.returncode:
            message = result.stderr.decode("utf-8", "replace").strip()
            if message == f"Error response from daemon: Could not find the file {target} in container {self.name}":
                return None
            raise OriginalReadError(f"Could not read {target} (docker cp exit {result.returncode}): {message}")
        try:
            with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
                members = archive.getmembers()
                if len(members) != 1 or not members[0].isfile() or members[0].name != target.rsplit("/", 1)[-1]:
                    raise OriginalReadError(f"Original {target} was not returned as one plain file")
                with archive.extractfile(members[0]) as file:
                    content = file.read()
                if len(content) != members[0].size:
                    raise OriginalReadError(f"Original {target} was not read completely")
                return content
        except (tarfile.TarError, OSError, EOFError) as error:
            raise OriginalReadError(f"Could not read original archive for {target}: {error}") from error

    def remove(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=30)


class Container:
    """One task container running one native session; records go to `folder`.

    The session runs with HOME=/root whatever the task image sets, so that the
    agent runtime uses an isolated home; some task images set HOME for a non-root
    user. Provider-specific startup belongs to WorkerContainer.
    """

    def __init__(self, image, folder, limits, workdir, environment, mounts):
        self.name = "delm-workspace-" + uuid.uuid4().hex[:12]
        self.folder = folder
        self.workdir = workdir
        environment = dict(environment)
        if limits['environment'].get('skills_dir'):
            environment['TASK_SKILLS_DIR'] = limits['environment']['skills_dir']
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "context-command-errors.jsonl").touch()
        self.compose = None
        if limits.get('compose_file'):
            from benchmark.task_compose import TaskCompose
            self.compose = TaskCompose(self.name, image, folder, limits, environment, mounts)
            (folder / 'launch.json').write_text(json.dumps(dict(container=self.name)))
            return
        command = ["docker", "run", "-d", "--name", self.name, "--label", "delm.workspace.research=true",
                   "--cpus", str(limits["environment"]["cpus"]), "--memory", f'{limits["environment"]["memory_mb"]}m',
                   "--pids-limit", "512", "--add-host", "host.docker.internal:host-gateway",
                   "--mount", f"type=bind,src={folder},dst=/records"]
        for source, target in mounts:
            command += ["--mount", f"type=bind,src={source},dst={target},readonly"]
        for key, value in environment.items():
            command += ["-e", f"{key}={value}"]
        command += [image, "sleep", "infinity"]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        (folder / "launch.json").write_text(json.dumps(dict(container=self.name)))

    def stop(self, *, grace_seconds=5):
        if self.compose:
            self.compose.stop()
            return
        subprocess.run(["docker", "stop", "-t", str(grace_seconds), self.name], capture_output=True, timeout=20)


def record_session_state(context, actor, state, lifecycle):
    """Record runner observations without interrupting the worker on a board error."""
    if context is not None:
        try:
            context.session_state(actor, state)
        except Exception as error:
            lifecycle.setdefault("session_state_error", f"{type(error).__name__}: {error}")


def watch(process, log, lifecycle, started, context=None, actor=None):
    """Record when the worker's lifecycle notices arrive on the runner's clock."""
    try:
        record_session_state(context, actor, "running", lifecycle)
        for line in process.stdout:
            log.write(line)
            log.flush()
            try:
                notice = json.loads(line)
            except ValueError:
                continue
            if isinstance(notice, dict) and notice.get("event") in ("model_start", "result", "deadline"):
                key = f"agent_{notice['event']}_received_seconds"
                lifecycle.setdefault(key, (time.monotonic_ns() - started) / 1e9)
    finally:
        process.wait()
        lifecycle["finished_seconds"] = (time.monotonic_ns() - started) / 1e9
        record_session_state(context, actor, "ended", lifecycle)
