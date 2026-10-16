"""Explicit external task inputs; execution stays in runtime.runner.

Only named files are snapshotted. Evaluation material is never sent to workers.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess


class ExternalTask:
    def __init__(self, path):
        path = Path(path).resolve()
        self.files = {"task.json": path.read_bytes()}
        self.config = json.loads(self.files["task.json"])
        c = self.config
        allowed = {"name", "image", "instruction", "files", "prompt", "rubric",
                   "cpus", "memory_mb", "storage_mb", "timeout_seconds",
                   "network", "timing", "evaluator"}
        if not isinstance(c, dict) or set(c) - allowed:
            raise ValueError("Unknown external task configuration fields")
        self.name = c.get("name", "")
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.name):
            raise ValueError("External task needs a simple name")
        if not isinstance(c.get("image"), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", c["image"]):
            raise ValueError("External task image must be an exact sha256 image ID")
        for key, default in (("cpus", 2), ("memory_mb", 4096), ("storage_mb", 1024), ("timeout_seconds", 180)):
            value = c.setdefault(key, default)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
            if key != "cpus" and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        for key, default, choices in (
                ("network", "model-only", ("model-only", "unrestricted")),
                ("timing", "synchronized", ("synchronized", "per-worker")),
                ("evaluator", "review", ("review",))):
            if c.setdefault(key, default) not in choices:
                raise ValueError(f"Unsupported {key}")

        def read(name):
            relative = safe_path(name)
            source = path.parent / relative
            if source.is_symlink() or not source.resolve().is_relative_to(path.parent) or not source.is_file():
                raise ValueError(f"Input must be a regular file inside the config directory: {name}")
            key = "inputs/" + relative.as_posix()
            self.files.setdefault(key, source.read_bytes())
            return self.files[key]

        self.instruction = read(c.get("instruction", "problem.txt")).decode()
        self.prompt = read(c["prompt"]).decode() if c.get("prompt") else None
        self.rubric = read(c["rubric"]) if c.get("rubric") else None
        self.initial_files = {}
        if not isinstance(c.get("files", {}), dict):
            raise ValueError("files must map workspace paths to input paths")
        for destination, source in c.get("files", {}).items():
            name = safe_path(destination).as_posix()
            if any(name.startswith(p + "/") or p.startswith(name + "/") for p in self.initial_files):
                raise ValueError("Initial file paths overlap")
            self.initial_files[name] = read(source)
        # Validate templates before creating containers or calling a model.
        if self.prompt:
            try:
                self.render("agent-1", ["agent-1", "agent-2"], "/workspace")
            except (KeyError, ValueError, IndexError, AttributeError) as error:
                raise ValueError(f"Invalid collaboration prompt template: {error}") from error

    def identity(self):
        return {"kind": "external", "files": {name: hashlib.sha256(raw).hexdigest()
                                                for name, raw in self.files.items()}}

    def limits(self):
        c = self.config
        return {"agent": {"timeout_sec": c["timeout_seconds"]},
                "environment": {k: c[k] for k in ("cpus", "memory_mb", "storage_mb")}}

    def render(self, actor, roster, workspace):
        return self.prompt.format(instruction=self.instruction, worker=actor,
                                  roster=", ".join(roster), count=len(roster), workspace=workspace)

    def snapshot(self, folder):
        for name, raw in self.files.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        for name, raw in self.initial_files.items():
            target = folder / "workspace" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)

    def seed(self, container, folder, workspace):
        if self.initial_files:
            subprocess.run(["docker", "cp", str(folder / "workspace") + "/.",
                            f"{container}:{workspace}/"],
                           check=True, capture_output=True, timeout=60)


def safe_path(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Expected a relative input path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in (".", "..", "") for p in name.split("/")):
        raise ValueError("Expected a relative input path without traversal")
    return path
