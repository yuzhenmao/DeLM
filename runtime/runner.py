"""Shared two/four-worker runtime; only roster and original prompts depend on n."""

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]

from runtime.config import DEFAULT_COUNT, PROTOCOLS
from runtime.machine_lock import machine_lock
from runtime.egress import RestrictedNetwork
from benchmark.external_task import ExternalTask
from reporting import review
from reporting.metrics import summarize
from runtime.containers import (OriginalFiles, board_address, collect_native_record,
                                handler, watch)
from benchmark.terminal_task import base_image_name, check_source, settings, verifier_image_name
from benchmark.images import inspect_image, validate_source
from benchmark.grading import prepare, evaluate
from board.protected_board import ProtectedBoard
from prompts.prompts_n4 import worker_prompt as prompt_n4
from prompts.prompts_n2 import worker_prompt as prompt_n2
from runtime import runtime_support as runtime

SERIES_FIELDS = ("commit", "source_sha256", "count", "model", "effort", "harness", "auth_mode",
                 "runtime_version", "authentication", "driver_revision", "service_cpus", "service_memory_mb", "timing_policy", "network_policy")


def worker_prompt(instruction, actor, roster, workdir):
    return {2: prompt_n2, 4: prompt_n4}[len(roster)](instruction, actor, roster, workdir)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source_files():
    packages = ("runtime", "board", "prompts", "benchmark", "reporting")
    return sorted([*ROOT.glob("*.py"),
                   *(path for package in packages for path in (ROOT / package).rglob("*.py")),
                   ROOT / "board/context.sh", ROOT / "Dockerfile.codex", ROOT / "Dockerfile.claude"])


def source_identity():
    """Require a clean Git checkout and record its commit and source hashes."""
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source_files()}
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"],
                                     cwd=ROOT, text=True)
    if status:
        raise RuntimeError("Commit the framework before launching; the checkout is not clean")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return commit, hashes


def capacity(limits, host_cpus, available_mb, service_cpus=0, service_memory_mb=0, count=DEFAULT_COUNT):
    """Sidecar reservations are per worker; never reduce official task quotas."""
    if limits.get("compose_file") and (service_cpus <= 0 or service_memory_mb <= 0):
        raise ValueError("Service tasks require an explicit CPU and memory reservation for each worker's services")
    cpus = count * (float(limits["environment"]["cpus"]) + service_cpus)
    memory = count * (int(limits["environment"]["memory_mb"]) + service_memory_mb)
    # Keep host capacity for the board, runtime, Docker, and collection.
    if host_cpus < cpus or available_mb < memory + 4096:
        raise RuntimeError(f"Need {cpus:g} task/service CPUs and {memory + 4096} MiB available RAM")
    return dict(task_and_service_cpus=cpus, task_and_service_memory_mb=memory,
                host_cpus=host_cpus, available_memory_mb=available_mb,
                service_cpus_per_worker=service_cpus, service_memory_mb_per_worker=service_memory_mb)


def preflight(task, auth, service_cpus, service_memory_mb, count=DEFAULT_COUNT, harness="codex", external=None):
    task_source = external.identity() if external else check_source(task)
    limits = external.limits() if external else settings(task)
    environment = limits["environment"]
    if environment.get("gpus", 0) or environment.get("mcp_servers") or not environment.get("allow_internet", True):
        raise ValueError("This runner supports unrestricted CPU task environments without MCP servers")
    if external and (service_cpus or service_memory_mb):
        raise ValueError("External tasks do not configure sidecar services")
    if not external and limits["verifier"].get("environment_mode") != "separate":
        raise ValueError("A separate official grader is required")
    if not external and list((task / "environment").glob("*compose*")):
        from benchmark.task_compose import SERVICE_BUILDS, service_images
        if task.name not in SERVICE_BUILDS:
            raise ValueError("The task's supplied services are not supported")
        limits["compose_file"] = str((task / "environment/docker-compose.yaml").resolve())
        limits["service_image_ids"] = {}
        for service, tag in service_images(task.name).items():
            service_image = inspect_image(tag)
            if service in SERVICE_BUILDS[task.name]:
                validate_source(service_image, task_source)
            limits["service_image_ids"][service] = service_image["Id"]
    runtime.validate_key_file(auth)
    if subprocess.check_output(["docker", "ps", "-q"], text=True, timeout=30).strip():
        raise RuntimeError("Use an idle machine; other Docker containers are running")
    if not Path("/proc/meminfo").exists():
        raise RuntimeError("Launch on Linux Docker Engine so capacity and worker clocks can be checked")
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    host = capacity(limits, len(os.sched_getaffinity(0)), int(memory["MemAvailable"].split()[0]) // 1024,
                    service_cpus, service_memory_mb, count)
    host["cpu_models"] = sorted({line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")})
    if shutil.disk_usage(ROOT).free < (count * environment["storage_mb"] + 4096) * 1024**2:
        raise RuntimeError("Insufficient free storage for all task workspaces and records")
    image = inspect_image(external.config["image"] if external else runtime.task_image(task, harness))
    verifier = None
    if not external:
        base = inspect_image(base_image_name(task))
        verifier = inspect_image(verifier_image_name(task))
        for info in (base, image, verifier):
            validate_source(info, task_source)
        runtime.validate_overlay(image, base, harness)
        limits["verifier"]["image_id"] = verifier["Id"]
    workdir = image["Config"].get("WorkingDir")
    if not workdir or workdir == "/":
        raise ValueError("Task image needs a non-root working directory")
    return task_source, limits, host, image, verifier


def capture_and_grade(task, containers, rows, limits, workdir, save):
    """Capture all live submissions, release their resources, then grade serially."""
    prepared = []
    removed = set()
    try:
        for container, row in zip(containers, rows):
            try:
                prepared.append((row, prepare(task, container.name, container.folder, limits, workdir)))
            except Exception as error:
                row["grading_error"] = f"Capture failed: {type(error).__name__}: {error}"
            save()
        for container in containers:
            stop_checked(container)
        for row, submission in prepared:
            try:
                row["grading"] = evaluate(submission)
                row["correct"] = row["grading"]["reward"] == 1
            except Exception as error:
                row["grading_error"] = f"{type(error).__name__}: {error}"
            finally:
                save()
                # A timed-out docker exec can leave the test running inside
                # its container. Remove it before starting the next grader.
                submission.remove()
                removed.add(submission.name)
    finally:
        errors = []
        for _, submission in prepared:
            if submission.name in removed:
                continue
            try:
                submission.remove()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("Prepared grader cleanup failed: " + "; ".join(errors))


def stop_checked(container):
    container.stop()
    names = [container.name]
    if container.compose:
        names = subprocess.check_output(["docker", "ps", "-aq", "--filter",
                                         "label=com.docker.compose.project=" + container.name], text=True, timeout=30).split()
        if not names:
            raise RuntimeError("Cannot locate the task's Compose containers after stop")
    inspected = json.loads(subprocess.check_output(["docker", "inspect", *names], timeout=30))
    if len(inspected) != len(names) or any(info["State"]["Running"] for info in inspected):
        raise RuntimeError("Task environments did not stop; refusing to grade beside them")


def remove_container(container):
    if container.compose:
        subprocess.run(container.compose.command + ["down", "--volumes"], check=True,
                       capture_output=True, timeout=60)
    else:
        subprocess.run(["docker", "rm", "-f", container.name], check=True, capture_output=True, timeout=30)


def create_container(*args, harness="codex", authentication="api-key"):
    # Retain the generated name even if Docker times out during construction.
    container = runtime.WorkerContainer.__new__(runtime.WorkerContainer)
    container.harness = harness
    container.authentication = authentication
    try:
        container.__init__(*args)
    except BaseException:
        if getattr(container, "name", None):
            if getattr(container, "compose", None):
                remove_container(container)
            elif args[2].get("compose_file") and (args[1] / "compose.override.json").exists():
                subprocess.run(["docker", "compose", "-p", container.name, "-f", args[2]["compose_file"],
                                "-f", str(args[1] / "compose.override.json"), "down", "--volumes"],
                               check=True, capture_output=True, timeout=60)
            else:
                subprocess.run(["docker", "rm", "-f", container.name], capture_output=True, timeout=30)
        raise
    return container


def create_original(image, workdir):
    original = OriginalFiles.__new__(OriginalFiles)
    try:
        original.__init__(image, workdir)
    except BaseException:
        if getattr(original, "name", None):
            subprocess.run(["docker", "rm", "-f", original.name], capture_output=True, timeout=30)
        raise
    return original


def read_worker_result(folder):
    """Keep damaged files as evidence without letting them interrupt cleanup."""
    try:
        value = json.loads((folder / "result.json").read_text())
        if not isinstance(value, dict):
            raise ValueError("Worker result is not an object")
        return value
    except (OSError, ValueError) as error:
        return dict(result_record_error=f"{type(error).__name__}: {error}")


def check_series(runs, task, identity):
    """A series fixes code and settings across its tasks; each task also fixes its images."""
    for path in sorted(runs.glob("*/settings.json")):
        recorded = json.loads(path.read_text())
        if (recorded != identity if path.parent.name == task
                else any(recorded.get(key) != identity[key] for key in SERIES_FIELDS)):
            raise ValueError("This series has different code, settings, or task images; choose a new --series")


@machine_lock()
def run(task, model, effort, series, repeat, auth, service_cpus=0, service_memory_mb=0, count=DEFAULT_COUNT, harness="codex", auth_mode="api-key", task_config=None):
    """Run one repeat; delm.py validates the arguments before calling this."""
    authentication = ("chatgpt" if harness == "codex" else "claude-oauth") if auth_mode == "coding-plan" else "api-key"
    adapter = runtime.runtime(harness)
    external = ExternalTask(task_config) if task_config else None
    task_name = external.name if external else task.name
    timing = external.config["timing"] if external else "per-worker"
    network_policy = external.config["network"] if external else "unrestricted"
    evaluator = external.config["evaluator"] if external else "official"
    commit, hashes = source_identity()
    task_source, limits, host, image, verifier = preflight(task, auth, service_cpus, service_memory_mb, count, harness, external)
    series_dir = ROOT / ".local" / "runs" / series / task_name
    identity = dict(commit=commit, source_sha256=hashes, task_source=task_source,
                    image_id=image["Id"], verifier_image_id=verifier["Id"] if verifier else None,
                    service_image_ids=limits.get("service_image_ids", {}),
                    count=count, model=model, effort=effort, harness=harness, auth_mode=auth_mode,
                    runtime_version=adapter.VERSION, authentication=authentication, driver_revision=adapter.DRIVER_REVISION,
                    service_cpus=service_cpus, service_memory_mb=service_memory_mb)
    identity.update(task_type="external" if external else "terminal-bench",
                    evaluator=evaluator, timing_policy=timing, network_policy=network_policy)
    check_series(series_dir.parent, task_name, identity)
    series_dir.mkdir(parents=True, exist_ok=True)
    if not (series_dir / "settings.json").exists():
        write_json(series_dir / "settings.json", identity)
    folder = series_dir / f"repeat-{repeat}"
    folder.mkdir()  # No overwrite or implicit restart.
    roster = [f"agent-{i}" for i in range(1, count + 1)]
    actors = {secrets.token_hex(16): actor for actor in roster}
    timeout = int(limits["agent"]["timeout_sec"])
    workdir = image["Config"]["WorkingDir"]
    image_path = next(v.split("=", 1)[1] for v in image["Config"]["Env"] if v.startswith("PATH="))
    metadata = dict(**identity, case=task_name, repeat=repeat, series=series, mode=f"delm-n{count}",
                    startup_protocol=PROTOCOLS[count], image_info=image, verifier_image_info=verifier,
                    task_settings=limits, host=host, timeout_seconds=timeout,
                    status="starting", phase="setup", agents=roster, workers=[],
                    started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    for source in source_files():
        destination = folder / "framework-source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    if external:
        external.snapshot(folder / "task-source")
    save = lambda: write_json(folder / "run.json", metadata)
    save()
    containers, processes = [], []
    context = server = original = restricted = None
    started = time.monotonic_ns()
    metadata["started_ns"] = started
    try:
        original = create_original(image["Id"], workdir)
        if external:
            external.seed(original.name, folder / "task-source", workdir)
        address = board_address()
        if network_policy == "model-only":
            restricted = RestrictedNetwork(harness)
            address = restricted.start()
            limits["network_name"] = restricted.name
        context = ProtectedBoard(folder / "context.sqlite", roster, original.fetch)
        server = ThreadingHTTPServer((address, 0), handler(context, actors))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://{address if restricted else 'host.docker.internal'}:{server.server_port}"
        for token, actor in actors.items():
            worker_folder = folder / actor
            environment = dict(WORKER_NAME=actor, TASK_WORKSPACE=workdir, AGENT_MODEL=model,
                               AGENT_EFFORT=effort, AGENT_TIMEOUT_SECONDS=timeout, AGENT_AUTHENTICATION=authentication,
                               CONTEXT_URL=url, CONTEXT_TOKEN=token,
                               CONTEXT_PYTHON="/opt/native/bin/python3.12", CONTEXT_LIBRARY_PATH="/opt/native/lib")
            environment["DELM_TIMING"] = timing
            if restricted:
                environment.update(restricted.environment())
            mounts = runtime.worker_mounts(harness, auth, authentication)
            container = create_container(image["Id"], worker_folder, limits, workdir, environment, mounts,
                                         harness=harness, authentication=authentication)
            containers.append(container)
            if external:
                external.seed(container.name, folder / "task-source", workdir)
            instruction = external.instruction if external else (task / "instruction.md").read_text()
            prompt = (external.render(actor, roster, workdir) if external and external.prompt
                      else worker_prompt(instruction, actor, roster, workdir))
            (worker_folder / "prompt.txt").write_text(prompt)
            container.preflight(limits["environment"]["storage_mb"], url)
        metadata.update(status="running", phase="execution")
        save()
        for actor, container in zip(roster, containers):
            process, log = container.start(image_path)
            lifecycle = dict(launch_seconds=(time.monotonic_ns() - started) / 1e9)
            watcher = threading.Thread(target=watch, args=(process, log, lifecycle, started, context, actor), daemon=True)
            processes.append((actor, container, process, log, watcher, lifecycle))
            watcher.start()
        if timing == "synchronized":
            ready_limit = time.monotonic() + 90
            while not all("agent_ready_received_seconds" in p[-1] for p in processes):
                if any(p.poll() is not None for _, _, p, _, _, _ in processes) or time.monotonic() >= ready_limit:
                    raise RuntimeError("Worker preparation failed; no task released")
                time.sleep(0.05)
            metadata["release_ns"] = time.monotonic_ns()
            metadata["deadline_ns"] = metadata["release_ns"] + timeout * 10**9
            for container in containers:
                write_json(container.folder / "release.json", {k: metadata[k] for k in ("release_ns", "deadline_ns")})
            save()
        print(json.dumps(dict(event="launched", folder=str(folder), workers=roster)), flush=True)
        while any(p.poll() is None for _, _, p, _, _, _ in processes):
            for _, container, process, _, _, lifecycle in processes:
                limit = (metadata["deadline_ns"] + 30 * 10**9 if timing == "synchronized"
                         else started + int((lifecycle["launch_seconds"] + timeout + 600) * 1e9))
                if process.poll() is None and time.monotonic_ns() > limit:
                    stop_checked(container)
                    metadata["runner_timeout"] = True
            time.sleep(1)
        metadata["phase"] = "collection"
        for actor, container, process, log, watcher, lifecycle in processes:
            watcher.join(timeout=20)
            if watcher.is_alive():
                raise RuntimeError("Worker lifecycle collection did not finish")
            log.close()
            row = read_worker_result(container.folder)
            row.update(actor=actor, process_returncode=process.returncode, **lifecycle)
            try:
                runtime.collect_record(harness, container, row, model)
            except Exception as error:
                row["record_error"] = f"{type(error).__name__}: {error}"
            row["context_command_transport_failures"] = len((container.folder / "context-command-errors.jsonl").read_text().splitlines())
            metadata["workers"].append(row)
            save()
        metadata["phase"] = "grading"
        save()
        if evaluator == "official":
            capture_and_grade(task, containers, metadata["workers"], limits, workdir, save)
        else:
            for container, row in zip(containers, metadata["workers"]):
                try:
                    review.capture(container, row, workdir)
                except Exception as error:
                    row["grading_error"] = f"Capture failed: {type(error).__name__}: {error}"
                save()
            for container in containers:
                stop_checked(container)
            review.export(folder, instruction, external.rubric, metadata["workers"])
        metadata["status"] = "finished"
        errors = ("record_error", "result_record_error", "transport_record_error", "session_state_error", "grading_error", "shutdown_error")
        if any(any(row.get(key) for key in errors) or row.get("grading", {}).get("workspace_copy_error")
               for row in metadata["workers"]):
            metadata["status"] = "incomplete"
        try:
            metadata["summary"] = summarize(metadata)
        except (KeyError, TypeError, ValueError) as error:
            metadata.update(status="incomplete", summary_error=str(error))
    except BaseException as error:
        metadata.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        def cleanup(label, operation):
            try:
                operation()
            except Exception as error:
                metadata["status"] = "incomplete"
                metadata.setdefault("cleanup_errors", []).append(f"{label}: {type(error).__name__}: {error}")
        for container in containers:
            cleanup("stop worker", lambda c=container: stop_checked(c))
        for _, _, process, log, watcher, _ in processes:
            cleanup("collect process exit", lambda p=process: p.wait(timeout=20))
            watcher.join(timeout=20)
            if not watcher.is_alive():
                log.close()
        # Preserve whatever the worker produced after an interrupted launch or
        # collection failure before removing its container.
        if metadata["status"] == "incomplete":
            from benchmark.terminal_task import extract_workspace
            for actor, container, process, _, _, lifecycle in processes:
                if not any(row.get("actor") == actor for row in metadata["workers"]):
                    row = read_worker_result(container.folder)
                    row.update(actor=actor, process_returncode=process.returncode, **lifecycle)
                    metadata["workers"].append(row)
            for container in containers:
                if harness == "codex" and not (container.folder / "native-trajectory.jsonl").exists():
                    cleanup("recover native record", lambda c=container: collect_native_record(c.name, c.folder))
                if not (container.folder / "workspace").exists():
                    def backup(c=container):
                        raw = subprocess.check_output(["docker", "cp", f"{c.name}:{workdir.rstrip('/')}/.", "-"],
                                                      stderr=subprocess.PIPE, timeout=120)
                        extract_workspace(raw, c.folder / "workspace")
                    cleanup("recover workspace", backup)
        if context:
            cleanup("stop board", context.stop)
        if server:
            cleanup("stop HTTP server", server.shutdown)
            cleanup("close HTTP server", server.server_close)
        if context:
            cleanup("export events", lambda: write_json(folder / "context-events.json", context.export()))
            cleanup("export attachments", lambda: write_json(folder / "context-bodies.json", context.export_bodies()))
        if original:
            cleanup("remove original container", original.remove)
        for container in containers:
            cleanup("remove worker environment", lambda c=container: remove_container(c))
        if restricted:
            cleanup("remove restricted network", restricted.close)
        if metadata["status"] != "finished":
            metadata.pop("summary", None)
        metadata.update(phase="finished", elapsed_seconds=(time.monotonic_ns() - started) / 1e9)
        save()
    print(json.dumps(dict(folder=str(folder), status=metadata["status"], summary=metadata.get("summary")), indent=2), flush=True)
    return metadata

