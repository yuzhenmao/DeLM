"""Board commands, run by an agent through its ordinary shell tool.

Files attached with --attach are read here and sent with the entry; apply
writes the files the board grants into the workspace. Paths are relative to
the task workspace, or absolute under /tmp.
"""

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

ACTIONS = ("read", "status", "claim", "reserve", "done", "note", "publish", "expand", "apply")
WAIT_CAP_SECONDS = 25           # the board's cap on one waiting read; the shell tool cuts a call off soon after 30 s
SCRATCH_ROOT = "/tmp"
RECORDS = Path("/records")      # the run's record folder: the prompt marks the session start, applied files are noted
FILE_LIMITS = dict(files=40, total_bytes=400_000, file_bytes=200_000)
VALIDATION_RETRIES = 20
TIMEOUT_SECONDS = 10
FILES_TIMEOUT_SECONDS = 90      # registering files makes the board fetch each original from the task image
SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build", ".cache", ".pytest_cache"}


class ClientError(ValueError):
    """A refused command; nothing was sent to the board."""


class ApplyError(ClientError):
    """A file-write failure, not a command validation failure."""


def request_context(payload, timeout=TIMEOUT_SECONDS):
    request = urllib.request.Request(
        os.environ["CONTEXT_URL"] + "/" + os.environ["CONTEXT_TOKEN"],
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as stream:
            return json.load(stream), True
    except HTTPError as error:
        if error.code == 400:
            return json.load(error), False
        if error.code == 429 or 500 <= error.code < 600:
            record_command_error(payload["action"], error)
        raise
    except (URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
        record_command_error(payload["action"], error)
        raise


def record_command_error(action, error, path=Path("/records/context-command-errors.jsonl")):
    record = dict(time_ns=time.monotonic_ns(), action=action, type=type(error).__name__,
                  reason=str(getattr(error, "reason", error)))
    with path.open("a") as output:
        output.write(json.dumps(record) + "\n")


# ---- files ------------------------------------------------------------------

def workspace():
    return Path(os.environ["TASK_WORKSPACE"]).resolve()


def locate(path, root):
    """The board's name for a path (relative to the workspace, or absolute under /tmp) and its place on disk."""
    given = Path(path)
    if ".." in given.parts:
        raise ClientError(f"{path}: path must not contain '..'")
    absolute = (given if given.is_absolute() else root / given)
    absolute = absolute.parent.resolve() / absolute.name        # directories resolved, the file itself not
    try:
        return absolute.relative_to(root).as_posix(), absolute
    except ValueError:
        pass
    try:
        return f"{SCRATCH_ROOT}/{absolute.relative_to(Path(SCRATCH_ROOT).resolve()).as_posix()}", absolute
    except ValueError:
        raise ClientError(f"{path}: only files in your workspace ({root}) or under {SCRATCH_ROOT} can be attached") from None


def bundle(paths):
    """Read the named files; every limit is checked here so nothing over it reaches the board."""
    root = workspace()
    if len(paths) > FILE_LIMITS["files"]:
        raise ClientError(f"{len(paths)} files given; at most {FILE_LIMITS['files']} can be attached")
    files, total, seen = [], 0, set()
    for path in paths:
        name, absolute = locate(path, root)
        if name in seen:
            raise ClientError(f"{path}: given twice")
        seen.add(name)
        if excluded_publication_file(name):
            raise ClientError(f"{path}: credentials, dependency/cache directories and grading records cannot be published")
        if absolute.is_symlink() or not absolute.is_file():
            raise ClientError(f"{path}: not a regular file\nResolved path: {absolute}\n"
                              f"Relative attachment paths use workspace root: {root}")
        raw = absolute.read_bytes()
        if len(raw) > FILE_LIMITS["file_bytes"]:
            raise ClientError(f"{path} is {len(raw):,} bytes; at most {FILE_LIMITS['file_bytes']:,} per file")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ClientError(f"{path}: not UTF-8 text; only text files can be attached") from None
        total += len(raw)
        files.append(dict(path=name, content=content, mode="x" if absolute.stat().st_mode & 0o111 else "-"))
    if total > FILE_LIMITS["total_bytes"]:
        raise ClientError(f"attached files total {total:,} bytes; at most {FILE_LIMITS['total_bytes']:,} allowed")
    return files


def excluded_publication_file(name):
    parts = Path(name).parts
    basename = Path(name).name.lower()
    return (bool(set(parts) & (SKIP_DIRS | {".codex", ".ssh", ".aws", ".config", ".claude", "records"}))
            or basename in {"auth.json", "credentials.json", "id_rsa", "id_ed25519", "grading.log", "reward.txt", "reward.json"}
            or basename == ".env" or basename.startswith(".env.")
            or name.startswith(("/tests/", "/solution/", "/records/")))


def publication_request_id(payload):
    """A repeated identical CLI publication reuses its ID, even after a lost response."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def local_hashes(names):
    """The sha256 of the caller's current copy of each attached path, or None when absent."""
    root = workspace()
    hashes = {}
    for name in names:
        _, absolute = locate(name, root)
        hashes[name] = hashlib.sha256(absolute.read_bytes()).hexdigest() if absolute.is_file() else None
    return hashes


def write_files(files, written=None):
    root = workspace()
    written = [] if written is None else written
    prepared, temporary_paths = [], []

    def temporary(parent):
        fd, name = tempfile.mkstemp(prefix=".delm-apply-", dir=parent)
        os.close(fd)
        path = Path(name)
        temporary_paths.append(path)
        return path

    try:
        for f in files:
            _, absolute = locate(f["path"], root)
            if absolute.exists() and not absolute.is_file():
                raise ClientError(f"{f['path']}: exists and is not a regular file")
            absolute.parent.mkdir(parents=True, exist_ok=True)
            backup = temporary(absolute.parent) if absolute.exists() else None
            if backup:
                shutil.copy2(absolute, backup, follow_symlinks=False)
            staged = temporary(absolute.parent)
            staged.write_text(f["content"], encoding="utf-8")
            mode = absolute.stat().st_mode if backup else 0o644
            staged.chmod(mode | 0o111 if f["mode"] == "x" else mode & ~0o111)
            prepared.append((f["path"], absolute, staged, backup))
        for name, absolute, staged, _ in prepared:
            os.replace(staged, absolute)
            written.append(name)
    except (OSError, ClientError) as error:
        failures = []
        for name, absolute, _, backup in reversed(prepared):
            if name not in written:
                continue
            try:
                if backup:
                    os.replace(backup, absolute)
                else:
                    absolute.unlink()
                written.remove(name)
            except OSError as rollback_error:
                if backup:
                    temporary_paths.remove(backup)  # keep the recoverable original
                failures.append(f"{name}: {rollback_error}; backup: {backup}")
        if failures:
            raise ClientError(f"{error}; rollback failed: {'; '.join(failures)}") from None
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    remember(files)


def known_record():
    """Content hashes of files this agent attached or applied, kept beside the run records."""
    path = RECORDS / "client-files.json"
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def remember(files):
    """Note files attached or applied, so later scans do not report them as unattached changes."""
    record = known_record()
    for f in files:
        record[f["path"]] = hashlib.sha256(f["content"].encode("utf-8")).hexdigest()
    try:
        (RECORDS / "client-files.json").write_text(json.dumps(record))
    except OSError:
        pass


def unattached_changes():
    """Workspace text files changed since the session started that this agent has neither attached nor applied.

    The prompt file's modification time marks the session start; nothing the
    agent edited can be older. A file edited again after it was attached is
    reported again. Large directories that are never work product are skipped.
    At most 20 names are reported.
    """
    marker = RECORDS / "prompt.txt"
    if not marker.exists():
        return []
    start, root, known, found = marker.stat().st_mtime, workspace(), known_record(), []
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            path = Path(folder) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                continue
            try:
                if path.stat().st_mtime <= start:
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            if len(raw) > FILE_LIMITS["file_bytes"]:
                continue
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if known.get(relative) == hashlib.sha256(raw).hexdigest():
                continue
            found.append(relative)
            if len(found) == 20:
                return found
    return found


# ---- commands ---------------------------------------------------------------

def render_expand(response):
    lines = [f"Files attached by {response['author']} to {response['id']}:"]
    for f in response["files"]:
        lines.append(f"\n--- {f['path']} ({f['bytes']:,} bytes, {f['change']}; shown {'in full' if f['shown'] == 'full' else 'as a diff against the original'})")
        if f.get("original_error"):
            lines.append("Original comparison unavailable: " + f["original_error"])
        lines.append(f["text"] if f["text"].endswith("\n") or not f["text"] else f["text"] + "\n")
    return "\n".join(lines)


def board_names(paths):
    """The board's names for paths given on the command line (relative to the workspace, or under /tmp)."""
    root = workspace()
    return [locate(path, root)[0] for path in paths]


def run_apply(args, only):
    listing, accepted = request_context(dict(action="attached", id=args.id))
    if not accepted:
        return listing, False
    names = only or [f["path"] for f in listing["files"]]
    response, accepted = request_context(dict(action="apply", id=args.id, local=local_hashes(names), paths=only))
    if not accepted or not response["applied"]:
        return response, False
    written = []
    try:
        write_files(response["files"], written)
    except (OSError, ClientError) as error:
        request_context(dict(action="apply-result", id=args.id, written=written, success=False))
        request_context(dict(action="apply-error", id=args.id, error=f"{type(error).__name__}: {error}"))
        raise ApplyError(f"the board granted the apply but writing failed: {error}") from None
    confirmed, _ = request_context(dict(action="apply-result", id=args.id, written=written, success=True))
    return dict(applied=True, author=response["author"], decisions=response["decisions"],
                written=[f["path"] for f in response["files"]],
                view=confirmed.get("view")), True


def wake_line(response):
    """One line the model reads first: what ended a waiting read, or that nothing did."""
    if not response.get("updates"):
        return f"No updates after {response.get('waited', 0):.0f} s."
    by = response.get("woke_by")
    if by == "already":
        return "The named entry is already on the board."
    events = response.get("woken") or []
    authors = sorted({e["actor"] for e in events})
    kinds = sorted({e["kind"] for e in events})
    return f"Update from {', '.join(authors) or 'the board'} after {response.get('waited', 0):.0f} s: {', '.join(kinds)}."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--all", action="store_true", help="read the full board, including previously seen entries")
    parser.add_argument("--wait", type=float, default=0.0, metavar="SECONDS",
                        help=f"read: wait at most {WAIT_CAP_SECONDS} seconds per call")
    parser.add_argument("--for", dest="target", default="", metavar="ID",
                        help="read --wait: the item or entry whose update you need")
    parser.add_argument("--text", default="")
    parser.add_argument("--item", default="")
    parser.add_argument("--subject", default="", help="reserve one item by its subject")
    parser.add_argument("--detail", default="", help="scope and required outcome of the reserved item")
    parser.add_argument("--summary", default="")
    parser.add_argument("--type", default="")
    parser.add_argument("--scope", default="")
    parser.add_argument("--basis", default="")
    parser.add_argument("--items", help="open implementation items as a JSON list")
    parser.add_argument("--id", default="", help="a done item's id or a note's id, for expand and apply")
    # The flag may be given once with several paths or repeated once per path; both accumulate.
    parser.add_argument("--attach", nargs="+", action="extend", default=None, metavar="PATH",
                        help="files to attach to a done or note")
    parser.add_argument("--only", nargs="+", action="extend", default=None, metavar="PATH",
                        help="only these attached files, for expand and apply")
    args = parser.parse_args()
    # Each agent has its own records directory. Keep corrected payloads in the
    # same retry budget, and serialize concurrent commands while updating it.
    key = json.dumps([args.action, args.item, args.id])
    if args.action == "reserve":
        key = json.dumps([args.action, args.subject])
    with (RECORDS / "context-validation-retries.json").open("a+") as state:
        fcntl.flock(state, fcntl.LOCK_EX)
        state.seek(0)
        failures = json.loads(state.read() or "{}")
        count = failures.get(key, 0)
        if count > VALIDATION_RETRIES:
            response, accepted = dict(error="Validation retry limit reached for this command target.",
                                     retries_remaining=0), False
        else:
            validation_error = False
            try:
                response, accepted = execute(args)
                validation_error = not accepted and "error" in response
            except ApplyError as error:
                response, accepted = dict(error=str(error)), False
            except ClientError as error:
                response, accepted = dict(error=str(error)), False
                validation_error = True
            if accepted:
                failures.pop(key, None)
            elif validation_error:
                failures[key] = count + 1
                response["retries_remaining"] = VALIDATION_RETRIES - count
            state.seek(0)
            state.truncate()
            state.write(json.dumps(failures))
    if args.action == "expand" and accepted:
        print(render_expand(response))
        if response.get("view") is not None:
            print("\nBoard changes:\n" + json.dumps(response["view"], ensure_ascii=False, indent=2))
    else:
        if args.action == "read" and accepted and args.wait:
            print(wake_line(response))
        print(json.dumps(response, ensure_ascii=False, indent=2))
    sys.exit(0 if accepted else 1)


def execute(args):
    attach, only = args.attach or [], args.only or []
    if args.items is not None and args.action != "publish":
        raise ClientError("--items goes with publish; reserve uses --subject and --detail")
    if (args.subject or args.detail) and args.action != "reserve":
        raise ClientError("--subject and --detail go with reserve")
    if args.action == "reserve" and args.item:
        raise ClientError("reserve uses --subject and --detail; claim an existing ID with claim --item")
    if args.action == "reserve" and args.items is not None:
        raise ClientError("--items goes with publish; reserve uses --subject and --detail")
    if attach and args.action not in ("done", "note"):
        raise ClientError("--attach goes with done or note")
    if only and args.action not in ("apply", "expand"):
        raise ClientError("--only goes with apply or expand")
    if args.id and args.action not in ("apply", "expand"):
        raise ClientError("--id goes with apply or expand; a queue item is named with --item")
    if args.target and args.action != "read":
        raise ClientError("--for goes with read")
    if args.wait and args.action != "read":
        raise ClientError("--wait goes with read")
    if args.wait < 0 or args.wait > WAIT_CAP_SECONDS:
        raise ClientError(f"--wait takes 1 to {WAIT_CAP_SECONDS} seconds; chain waits for longer")
    only = board_names(only)
    if args.action == "apply":
        response, accepted = run_apply(args, only)
    elif args.action == "expand":
        response, accepted = request_context(dict(action="expand", id=args.id, paths=only))
    elif args.action == "read":
        payload = dict(action="read", all=args.all, wait=args.wait)
        if args.target:
            payload["for"] = args.target
        response, accepted = request_context(payload, TIMEOUT_SECONDS + args.wait)
    else:
        payload = {k: v for k, v in vars(args).items()
                   if k not in ("attach", "only", "id", "wait", "target")}
        if args.items is not None:
            try:
                payload["items"] = json.loads(args.items)
            except ValueError as error:
                raise ClientError(f"--items must be valid JSON: {error}") from None
        if attach:
            payload["files"] = bundle(attach)
        if args.action in ("note", "done"):
            payload["request_id"] = publication_request_id(payload)
        response, accepted = request_context(payload, FILES_TIMEOUT_SECONDS if attach else TIMEOUT_SECONDS)
        if accepted and attach:
            remember(payload["files"])
            missing = unattached_changes()
            if missing:
                response["changed_but_not_attached"] = missing
    return response, accepted


if __name__ == "__main__":
    main()
