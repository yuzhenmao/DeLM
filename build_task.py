"""Build the official task image, then the selected provider's independent runtime."""

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from benchmark.terminal_task import base_image_name, check_source, ensure_verifier_image
from benchmark.task_compose import build_services
from benchmark.images import inspect_image, source_label
from runtime.config import CLI_VERSIONS, DRIVER_REVISION
from runtime.runtime_support import task_image


ROOT = Path(__file__).resolve().parent


def build(folder, task, harness="codex"):
    if harness not in ("codex", "claude-code"):
        raise ValueError("Unsupported harness")
    folder.mkdir(parents=True, exist_ok=True)
    image = task_image(task, harness)
    base = base_image_name(task)
    started = time.monotonic()
    metadata = dict(image=image, harness=harness, base_image=base, returncode=1, stages={})
    stage = "source"
    try:
        source = check_source(task)
        metadata["task_source"] = source
        label = source_label(source)
        metadata["stages"][stage] = dict(returncode=0)
        stage = "base"
        base_dockerfile = folder / "Dockerfile.task"
        shutil.copyfile(task / "environment/Dockerfile", base_dockerfile)
        dockerfile = folder / "Dockerfile"
        provider = "codex" if harness == "codex" else "claude"
        shutil.copyfile(ROOT / f"Dockerfile.{provider}", dockerfile)
        # Disable fresh attestations so cached builds retain the same base ID.
        subprocess.run(["docker", "build", "--progress=plain", "--provenance=false",
                        "--label", label, "-t", base, "-f", str(base_dockerfile),
                        str(task / "environment")], check=True)
        base_info = inspect_image(base)
        metadata["base_image_info"] = base_info
        metadata["stages"][stage] = dict(returncode=0, image_id=base_info["Id"])
        stage = "runtime"
        subprocess.run(["docker", "build", "--progress=plain", "--provenance=false", "--target", "task",
                        "--build-arg", f"TASK_IMAGE={base}",
                        "--build-arg", f"CLI_VERSION={CLI_VERSIONS[harness]}",
                        "--build-arg", f"DRIVER_REVISION={DRIVER_REVISION}",
                        "--label", label, "--label", f"delm.harness={harness}",
                        "--label", f"delm.runtime.version={CLI_VERSIONS[harness]}",
                        "--label", f"delm.driver={DRIVER_REVISION}",
                        "--label", f"delm.base-task-image={base_info['Id']}",
                        "-t", image, "-f", str(dockerfile), str(ROOT)], check=True)
        metadata["image_info"] = inspect_image(image)
        metadata["stages"][stage] = dict(returncode=0, image_id=metadata["image_info"]["Id"])
        stage = "verifier"
        verifier = ensure_verifier_image(task, source=source)
        metadata["stages"][stage] = dict(returncode=0, image=verifier,
                                         image_id=inspect_image(verifier)["Id"])
        stage = "services"
        services = build_services(task, folder / "services", source=source)
        metadata["stages"][stage] = dict(returncode=0,
            image_ids={name: inspect_image(tag)["Id"] for name, tag in services.items()})
        metadata["returncode"] = 0
    except Exception as error:
        code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
        metadata["returncode"] = code if code > 0 else 128 - code
        metadata["stages"][stage] = dict(returncode=metadata["returncode"],
                                         error=f"{type(error).__name__}: {error}")
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - started
        (folder / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata["returncode"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("--harness", choices=("codex", "claude-code"), default="codex")
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    if args.folder:
        raise SystemExit(build(args.folder, args.task.resolve(), args.harness))
    folder = ROOT / "evidence" / ("build-" + args.harness + "-" + args.task.name + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    with (folder / "build.log").open("x") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(args.task.resolve()),
                                    "--harness", args.harness, "--folder", str(folder)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    (folder / "launch.json").write_text(json.dumps(dict(pid=process.pid)))
    print(json.dumps(dict(folder=str(folder), pid=process.pid)))
