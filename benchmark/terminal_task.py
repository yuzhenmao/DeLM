"""Task source checks and post-agent grading; never part of the shared context."""

import hashlib
import io
import json
import re
import shutil
import stat
import subprocess
import tarfile
import time
import tomllib
from pathlib import Path

from benchmark.images import source_label


def settings(task):
    return tomllib.loads((task / "task.toml").read_text())


def check_source(task):
    root = task if (task / "source.json").exists() else task.parent
    manifest = json.loads((root / "source.json").read_text())
    for record in manifest["files"]:
        raw = (root / record["path"]).read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Downloaded task file changed: {record['path']}")
    return manifest


def image_name(task):
    return f"delm-workspace-{task.name}:local"


def base_image_name(task):
    return f"delm-task-{task.name}:base"


def verifier_image_name(task):
    return f"delm-verify-{task.name}:local"


def ensure_verifier_image(task, timeout=3600, *, source=None):
    """Build the task's own verifier environment (tests/Dockerfile), unchanged."""
    image = verifier_image_name(task)
    label = source_label(check_source(task) if source is None else source)
    # Grading happens after the agents finish, so an emulated verifier never
    # touches agent timing; it is used only when the task's verifier cannot be
    # built for this machine's architecture.
    for platform in ([], ["--platform", "linux/amd64"]):
        result = subprocess.run(["docker", "build", *platform, "--provenance=false", "--label", label, "-t", image, str(task / "tests")],
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return image
    raise RuntimeError(f"Verifier image build failed for {task.name}: "
                       + result.stdout[-2000:] + result.stderr[-2000:])


ENVIRONMENT_FAILURE = re.compile(
    r"No module named|ModuleNotFoundError|command not found|Cannot find module|"
    r"No such file or directory: '?/tests")


def grading_result(folder):
    """Validate the task verifier's own verdict.

    Verifiers are per-task (pytest with a CTRF report, vitest, gate scripts);
    Read reward.txt, or the reward field in reward.json when text is absent.
    When a CTRF report exists it must agree with the reward; otherwise the verdict
    stands unless the grading log shows the verifier environment itself failed.
    """
    reward_file = folder / "reward.txt"
    if reward_file.exists():
        reward = float(reward_file.read_text())
    elif (folder / "reward.json").exists():
        try:
            value = json.loads((folder / "reward.json").read_text())["reward"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError("Invalid JSON reward from official grader") from error
        if type(value) not in (int, float):
            raise RuntimeError("JSON reward must be a number")
        reward = float(value)
    else:
        raise RuntimeError("Official grader produced no reward")
    if reward not in (0, 1):
        raise RuntimeError("Reward is not a 0/1 task score")
    report = folder / "ctrf.json"
    if not report.exists():
        log = folder.parent / "grading.log"
        if log.exists() and not ENVIRONMENT_FAILURE.search(log.read_text()):
            return dict(reward=reward, tests=None)
        raise RuntimeError("Verifier environment failure; reward is not a valid task score")
    summary = json.loads(report.read_text())["results"]["summary"]
    tests = summary["tests"]
    if (tests <= 0 or summary["passed"] + summary["failed"] != tests or
            any(summary.get(key, 0) for key in ("skipped", "pending", "other"))):
        raise RuntimeError("Grader did not complete all collected tests")
    if reward != int(summary["failed"] == 0):
        raise RuntimeError("Reward and pytest test report disagree")
    return dict(reward=reward, tests=summary)


def artifact_container(container, service):
    """Resolve a declared service only within this agent's compose project."""
    if not service or service == 'main':
        return container
    info = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]
    project = (info['Config'].get('Labels') or {}).get('com.docker.compose.project')
    if not project:
        raise RuntimeError(f'Artifact service {service!r} requires a compose project')
    matches = subprocess.check_output([
        'docker', 'ps', '-a', '--filter', f'label=com.docker.compose.project={project}',
        '--filter', f'label=com.docker.compose.service={service}', '--format', '{{.ID}}'], text=True).split()
    if len(matches) != 1:
        raise RuntimeError(f'Expected one artifact service {service!r} in {project}, found {len(matches)}')
    return matches[0]


def copy_artifacts(container, artifacts, verifier, staging, recording=None, *, record_main=False):
    """Mirror the task's declared artifact upload into the verifier container.

    Only declared paths cross over, so verifier-baked canonical data cannot be
    overwritten by undeclared agent files. A missing artifact is recorded and
    skipped; its absence is the verifier's to judge.
    """
    Path(staging).mkdir(parents=True, exist_ok=True)
    missing = []
    saved = []
    for index, artifact in enumerate(artifacts):
        source = artifact['source'] if isinstance(artifact, dict) else artifact
        service = artifact.get('service') if isinstance(artifact, dict) else None
        origin = artifact_container(container, service)
        target = source.rstrip("/") or "/"
        # Declared directories do not always carry a trailing slash, and the
        # verifier may already contain the directory; a directory copied onto an
        # existing one would nest inside it, so directories are always copied
        # by content.
        is_dir = subprocess.run(["docker", "exec", origin, "test", "-d", target],
                                capture_output=True, timeout=15).returncode == 0
        stage = Path(staging) / f"artifact-{index}"
        if is_dir:
            stage.mkdir()
        copied = subprocess.run(["docker", "cp", f"{origin}:{target}" + ("/." if is_dir else ""), str(stage)],
                                capture_output=True, timeout=120).returncode == 0
        if not copied:
            missing.append(artifact)
            continue
        if recording is not None and (record_main or service and service != 'main'):
            recording = Path(recording)
            recording.mkdir(parents=True, exist_ok=True)
            destination = recording / f'artifact-{index}'
            if is_dir:
                shutil.copytree(stage, destination, symlinks=True)
            else:
                shutil.copy2(stage, destination)
            saved.append(dict(service=service or 'main', source=source, file=destination.name))
            (recording / 'manifest.json').write_text(json.dumps(saved, indent=2))
        subprocess.run(["docker", "exec", verifier, "mkdir", "-p", str(Path(target).parent)],
                       check=True, capture_output=True, timeout=15)
        if is_dir:
            subprocess.run(["docker", "exec", verifier, "mkdir", "-p", target],
                           check=True, capture_output=True, timeout=15)
            subprocess.run(["docker", "cp", str(stage) + "/.", f"{verifier}:{target}"],
                           check=True, capture_output=True, timeout=120)
        else:
            # Harbor makes a file artifact's parent writable in the verifier.
            subprocess.run(["docker", "exec", verifier, "chmod", "777", str(Path(target).parent)],
                           check=True, capture_output=True, timeout=15)
            subprocess.run(["docker", "cp", str(stage), f"{verifier}:{target}"],
                           check=True, capture_output=True, timeout=120)
    return missing


def run_collect_hooks(container, hooks, folder):
    """Run the task's official collection commands after agent completion.

    Match the task format's default 60-second timeout and best-effort execution.
    The official verifier decides whether the captured artifacts are usable.
    """
    if not hooks:
        return []
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for index, hook in enumerate(hooks):
        record = dict(hook)
        started = time.monotonic()
        with (folder / f'{index}.log').open('w') as log:
            try:
                origin = artifact_container(container, hook.get('service'))
                command = ['docker', 'exec']
                if hook.get('user') is not None:
                    command += ['--user', str(hook['user'])]
                command += [origin, 'sh', '-c', hook['command']]
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=hook.get('timeout_sec', 60))
                record['returncode'] = result.returncode
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
                log.write(record['error'] + '\n')
        record['elapsed_seconds'] = time.monotonic() - started
        records.append(record)
        (folder / 'results.json').write_text(json.dumps(records, indent=2))
    return records


def collect_artifacts(container, limits, verifier, worker, workspace):
    """Capture main artifacts before stopping main for sidecar collection."""
    artifacts = limits.get('artifacts', [workspace])
    hooks = limits.get('verifier', {}).get('collect', [])
    staging = worker / 'artifact-staging'
    if not hooks:
        return copy_artifacts(container, artifacts, verifier, staging, worker / 'service-artifacts')
    is_main = lambda entry: not isinstance(entry, dict) or entry.get('service') in (None, 'main')
    main_hooks = [h for h in hooks if h.get('service', 'main') == 'main']
    sidecar_hooks = [h for h in hooks if h.get('service', 'main') != 'main']
    main_artifacts = [a for a in artifacts if is_main(a)]
    sidecar_artifacts = [a for a in artifacts if not is_main(a)]
    run_collect_hooks(container, main_hooks, worker / 'artifact-collection/main')
    missing = copy_artifacts(container, main_artifacts, verifier, staging / 'main',
                             worker / 'collected-artifacts/main', record_main=True)
    if sidecar_hooks or sidecar_artifacts:
        # Separate-verifier collection freezes agent processes before a service
        # snapshot; service containers remain running in this worker's project.
        subprocess.run(['docker', 'stop', '-t', '5', container], check=True,
                       capture_output=True, timeout=30)
    run_collect_hooks(container, sidecar_hooks, worker / 'artifact-collection/services')
    missing += copy_artifacts(container, sidecar_artifacts, verifier, staging / 'services',
                              worker / 'collected-artifacts/services', record_main=True)
    return missing


def record_filter(member, path):
    """How the workspace archive is unpacked for the record.

    The `tar` filter keeps every path inside the folder and keeps links as the agent
    made them (an absolute link is common: `ln -s /app/work/x analysis/x`). On top of it,
    directories take default modes, so a task image with read-only directories
    cannot stop the copy as it stops `docker cp` on Linux, and files stay readable by the owner, so the record can be
    collected.
    """
    member = tarfile.tar_filter(member, path)
    if member.isdir():
        return member.replace(mode=None)
    if member.isreg() and member.mode is not None:
        return member.replace(mode=member.mode | stat.S_IRUSR | stat.S_IWUSR)
    return member


def extract_workspace(archive, folder):
    """Unpack the `docker cp` archive of a container's workspace into `folder`, for the record."""
    folder.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(folder, filter=record_filter)
