"""Capture every submission before releasing environments and grading them.

Task collection hooks, artifact rules, and reward parsing are shared with
terminal_task.py. Prepared grader containers sleep while files are copied and
are stopped until evaluation, so no tests run alongside the task environments.
"""

import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from benchmark.terminal_task import collect_artifacts, extract_workspace, grading_result, verifier_image_name


def docker(*args, timeout=60, **kwargs):
    return subprocess.run(["docker", *map(str, args)], timeout=timeout, **kwargs)


@dataclass
class PreparedGrade:
    name: str
    worker: Path
    timeout: int
    missing: list
    workspace_copy_error: str | None

    def remove(self):
        docker("rm", "-f", self.name, capture_output=True, timeout=30, check=True)


def prepare(task, container, worker, limits, workspace):
    """Capture live state without evaluating it; preserve all staging records."""
    verifier_settings = limits["verifier"]
    if verifier_settings.get("environment_mode") != "separate":
        raise ValueError("Only tasks with a separate official grader are supported")
    copy_error = None
    try:
        archive = docker("cp", f"{container}:{workspace.rstrip('/')}/.", "-",
                         capture_output=True, check=True, timeout=120).stdout
        extract_workspace(archive, worker / "workspace")
    except Exception as error:
        copy_error = f"{type(error).__name__}: {error}"
    name = "delm-grade-" + uuid.uuid4().hex[:12]
    command = ["run", "-d", "--name", name, "--label", "delm.workspace.research=true"]
    for key, value in verifier_settings.get("env", {}).items():
        command += ["-e", f"{key}={value}"]
    resources = verifier_settings.get("environment", {})
    if resources.get("allow_internet") is False:
        command += ["--network", "none"]
    if "cpus" in resources:
        command += ["--cpus", str(resources["cpus"])]
    if "memory_mb" in resources:
        command += ["--memory", f"{resources['memory_mb']}m"]
    try:
        docker(*command, verifier_settings.get("image_id", verifier_image_name(task)), "sh", "-c", "sleep infinity",
               capture_output=True, check=True)
        missing = collect_artifacts(container, limits, name, worker, workspace)
        docker("stop", "-t", "1", name, check=True, capture_output=True, timeout=20)
        return PreparedGrade(name, worker, int(verifier_settings["timeout_sec"]), missing, copy_error)
    except BaseException:
        # Docker may have created it even if the client timed out.
        docker("rm", "-f", name, capture_output=True, timeout=30)
        raise


def evaluate(prepared):
    """Run the unchanged official command after all task environments stop."""
    started = time.monotonic()
    returncode = None
    docker("start", prepared.name, check=True, capture_output=True)
    docker("exec", prepared.name, "mkdir", "-p", "/logs/verifier", check=True, capture_output=True)
    docker("exec", prepared.name, "chmod", "777", "/logs/verifier", check=True, capture_output=True)
    try:
        with (prepared.worker / "grading.log").open("x") as log:
            try:
                returncode = docker("exec", prepared.name, "bash", "/tests/test.sh",
                                    stdout=log, stderr=subprocess.STDOUT, timeout=prepared.timeout).returncode
            except subprocess.TimeoutExpired:
                pass
    finally:
        # Killing the docker exec client does not kill its container process.
        # Stop it and any background children before copying stable logs.
        docker("stop", "-t", "1", prepared.name, capture_output=True, check=True, timeout=20)
    docker("cp", f"{prepared.name}:/logs/verifier", prepared.worker / "grading",
           capture_output=True, check=True, timeout=30)
    outcome = dict(returncode=returncode, missing_artifacts=prepared.missing,
                   elapsed_seconds=time.monotonic() - started,
                   workspace_copy_error=prepared.workspace_copy_error)
    if returncode is not None:
        return dict(**grading_result(prepared.worker / "grading"), **outcome)
    try:
        verdict = grading_result(prepared.worker / "grading")
    except RuntimeError:
        verdict = dict(reward=0.0, tests=None)
    return dict(**verdict, verifier_timeout=prepared.timeout, **outcome)
