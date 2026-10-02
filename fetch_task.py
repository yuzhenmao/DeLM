"""Download one official task without changing its contents."""

import hashlib
import argparse
import json
import urllib.request
from pathlib import Path, PurePosixPath


REPOSITORY = "harbor-framework/terminal-bench"
ROOT = Path(__file__).resolve().parent / "tasks" / "terminal-bench-4.0"


def request(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "DeLM-Framework-research"}),
        timeout=60,
    ).read()


def fetch(task, revision):
    if not task or PurePosixPath(task).name != task or task in (".", ".."):
        raise ValueError("Expected a task name, not a path")
    root = ROOT / task
    if root.exists():
        raise SystemExit("Task directory already exists; nothing replaced")
    api = f"https://api.github.com/repos/{REPOSITORY}"
    revision = json.loads(request(f"{api}/commits/{revision}"))["sha"]
    tree = json.loads(request(f"{api}/git/trees/{revision}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("Incomplete source tree")
    prefix = f"tasks/{task}/"
    files = [item for item in tree["tree"] if item["type"] == "blob" and
             (item["path"].startswith(prefix) or item["path"] == "LICENSE")]
    if not any(item["path"] == prefix + "instruction.md" for item in files):
        raise RuntimeError("Task instruction missing")
    records = []
    for item in files:
        relative = item["path"].removeprefix(prefix)
        if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise RuntimeError("Invalid source path")
        destination = root / relative
        raw = request(f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{item['path']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output:
            output.write(raw)
        records.append(dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        print(relative, len(raw), flush=True)
    manifest = dict(repository=f"https://github.com/{REPOSITORY}", revision=revision,
                    task=task, files=records)
    (root / "source.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(revision=revision, files=len(records), bytes=sum(x["bytes"] for x in records))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    fetch(args.task, args.revision)
