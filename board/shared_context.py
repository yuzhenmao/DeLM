"""The shared context: one append-only event log, written and read by agents.

Entries: status, item, claim, done, note (FACT or FAIL), runner-only session state, and a
read record for every view served. A done or note may register files behind
it (a body); the read shows only their names and sizes, and a peer pulls the
content by id with expand or apply. Every pull is recorded with what was
served or written. The only way an agent learns the board is to read it.

Bodies are task-derived content, so they live in a second database beside
the log (`context-bodies.sqlite`), attached to the same connection so that an
entry and its files are written in one transaction.
"""

import difflib
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

FILE_LIMITS = dict(files=40, total_bytes=400_000, file_bytes=200_000)
SCRATCH_ROOT = "/tmp"           # attached paths are relative to the workspace, or absolute under this root
NOTE_TYPES = ("FACT", "FAIL")
SYSTEM = "system"
# A waiting read is one blocking shell command; the native harness cuts a tool call
# off after about 31 s, so the board never holds one longer than this.
WAIT_CAP_SECONDS = 25
WAIT_POLL_SECONDS = 0.25
# Peer entries that end a waiting read; a status never does (see wake_reason).
WAKE_KINDS = ("item", "note", "body", "done")


class BoardError(ValueError):
    """A rejected write; the message is returned to the agent verbatim."""


class OriginalReadError(RuntimeError):
    """The original task file could not be read; its absence is not established."""


def limited(field, value):
    """Validate required text without enforcing the prompt's brevity targets."""
    if not isinstance(value, str) or not value.strip():
        raise BoardError(f"{field} must be non-empty text")
    return value


def normalized(subject):
    """Ignore case and extra whitespace, preserving punctuation and Unicode text."""
    return " ".join(subject.lower().split())


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check_path(path):
    """A registered path: relative to the workspace, or absolute under the scratch root; no `..`."""
    if not isinstance(path, str) or not path or path != path.strip():
        raise BoardError("every attached file needs a path")
    pure = PurePosixPath(path)
    if ".." in pure.parts or pure.name in ("", ".", ".."):
        raise BoardError(f"{path}: path must not contain '..'")
    if pure.is_absolute() and not path.startswith(SCRATCH_ROOT + "/"):
        raise BoardError(f"{path}: absolute paths are accepted only under {SCRATCH_ROOT}; use a path relative to your workspace")
    return path


def check_files(files):
    """Validate a bundle the client sent: [{path, content, mode}] with the limits named on failure."""
    if not isinstance(files, list) or not files:
        raise BoardError("files must be a non-empty list")
    if len(files) > FILE_LIMITS["files"]:
        raise BoardError(f"{len(files)} files attached; at most {FILE_LIMITS['files']} allowed")
    total, seen, checked = 0, set(), []
    for entry in files:
        path = check_path(entry.get("path") if isinstance(entry, dict) else None)
        if path in seen:
            raise BoardError(f"{path}: attached twice")
        seen.add(path)
        content = entry.get("content")
        if not isinstance(content, str):
            raise BoardError(f"{path}: content must be text")
        raw = content.encode("utf-8")
        if len(raw) > FILE_LIMITS["file_bytes"]:
            raise BoardError(f"{path} is {len(raw):,} bytes; at most {FILE_LIMITS['file_bytes']:,} per file")
        total += len(raw)
        checked.append(dict(path=path, content=content, bytes=len(raw), sha256=sha256(raw),
                            mode="x" if entry.get("mode") == "x" else "-"))
    if total > FILE_LIMITS["total_bytes"]:
        raise BoardError(f"attached files total {total:,} bytes; at most {FILE_LIMITS['total_bytes']:,} allowed")
    return checked


def metadata(files):
    return [dict(path=f["path"], bytes=f["bytes"], change=f["change"],
                 **({"original_error": f["original_error"]} if f.get("original_error") else {})) for f in files]


class SharedContext:
    def __init__(self, path, actors, original=None):
        """`original(path)` returns bytes or confirmed absence (None), or raises OriginalReadError."""
        self.path = Path(path)
        self.bodies_path = self.path.with_name("context-bodies.sqlite")
        self.actors = tuple(actors)
        self.original = original or (lambda path: None)
        self.stopping = threading.Event()       # set at shutdown so waiting reads return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, time_ns INTEGER NOT NULL,
                actor TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS bodydb.bodies (
                id TEXT PRIMARY KEY, time_ns INTEGER NOT NULL, actor TEXT NOT NULL, files TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS bodydb.publications (
                id TEXT PRIMARY KEY, actor TEXT NOT NULL, kind TEXT NOT NULL,
                item_id TEXT, payload TEXT NOT NULL, fingerprint TEXT NOT NULL,
                request_id TEXT, UNIQUE(actor, request_id))""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("ATTACH DATABASE ? AS bodydb", (str(self.bodies_path),))
        try:
            with db:
                db.execute("BEGIN IMMEDIATE")
                yield db
        finally:
            db.close()

    @staticmethod
    def append(db, actor, kind, payload):
        db.execute("INSERT INTO events(time_ns, actor, kind, payload) VALUES (?, ?, ?, ?)",
                   (time.monotonic_ns(), actor, kind, json.dumps(payload, ensure_ascii=False)))

    @staticmethod
    def events(db, *kinds):
        marks = ",".join("?" * len(kinds))
        return [(row["actor"], row["kind"], json.loads(row["payload"])) for row in db.execute(
            f"SELECT actor, kind, payload FROM events WHERE kind IN ({marks}) ORDER BY sequence", kinds)]

    # ---- derived state -------------------------------------------------

    def queue_state(self, db):
        """Derive work ownership without a planning gate or privileged planner."""
        items = {}
        for actor, kind, payload in self.events(db, "item", "claim", "done"):
            if kind == "item":
                items[payload["id"]] = dict(payload, state="open")
            elif kind == "claim":
                items[payload["item"]].update(state="claimed", owner=actor)
            elif payload["item"] in items:
                items[payload["item"]].update(state="done", summary=payload["summary"], owner=actor,
                                              publication_id=payload.get("publication_id", payload["item"]))
        return list(items.values())

    @staticmethod
    def last_sequence(db):
        return db.execute("SELECT MAX(sequence) FROM events").fetchone()[0] or 0

    @staticmethod
    def last_read(db, actor):
        return db.execute("SELECT MAX(sequence) FROM events WHERE actor = ? AND kind = 'read'", (actor,)).fetchone()[0]

    @staticmethod
    def current_status(db, actor):
        row = db.execute("SELECT payload FROM events WHERE actor = ? AND kind = 'status' ORDER BY sequence DESC LIMIT 1",
                         (actor,)).fetchone()
        return json.loads(row["payload"])["text"] if row else None

    def view(self, db, reader):
        current = {}
        for actor, _, payload in self.events(db, "status"):
            current[actor] = payload["text"]
        sessions = dict.fromkeys(self.actors, "starting")
        for _, _, payload in self.events(db, "session"):
            sessions[payload["actor"]] = payload["state"]
        queue = self.queue_state(db)
        attached = {payload["id"]: payload["files"] for _, _, payload in self.events(db, "body")}
        for item in queue:
            publication_id = item.get("publication_id", item["id"])
            if publication_id in attached:
                item["files"] = attached[publication_id]
        notes = [dict(payload, actor=actor) for actor, _, payload in self.events(db, "note")]
        for note in notes:
            if note["id"] in attached:
                note["files"] = attached[note["id"]]
        return dict(reader=reader, agents=list(self.actors),
                    sessions=[dict(actor=a, state=sessions[a]) for a in self.actors],
                    status=[dict(actor=a, text=current[a]) for a in self.actors if a in current],
                    queue=queue, notes=notes,
                    sequence=self.last_sequence(db))

    @staticmethod
    def compact(item):
        """An active item the reader has already seen in full: its identity and state, not its description."""
        return {k: v for k, v in item.items() if k in ("id", "subject", "state", "owner", "files", "publication_id")}

    def served(self, db, actor, via, full=False):
        """Active work plus changes since this actor's last served view.

        Every later call re-reads what the model was shown, so an unchanged
        active item is listed by id, subject, state and owner only; its detail
        was served when it appeared and `read --all` shows it again. A read or
        a served wait lists the whole active queue that way; a write's response
        carries only the items that changed.
        """
        result = self.view(db, actor)
        previous = self.last_read(db, actor)
        result["full"] = full or previous is None
        if not result["full"]:
            changed, statuses = set(), set()
            for row in db.execute(
                "SELECT actor, kind, payload FROM events WHERE sequence > ?", (previous,)
            ):
                payload = json.loads(row["payload"])
                if row["kind"] in ("item", "note", "body"):
                    changed.add(payload["id"])
                elif row["kind"] in ("claim", "done"):
                    changed.add(payload["item"])
                elif row["kind"] == "status":
                    statuses.add(row["actor"])
            listed = via in ("read", "wait")
            result["queue"] = [i if i["id"] in changed else self.compact(i)
                               for i in result["queue"] if i["id"] in changed or (listed and i["state"] != "done")]
            result["notes"] = [n for n in result["notes"] if n["id"] in changed]
            result["status"] = [s for s in result["status"] if s["actor"] in statuses]
        self.append(db, actor, "read", dict(view=result, via=via))
        return result

    def view_after(self, actor, via):
        """The incremental view served with a write's response, so no separate read follows it."""
        with self.connect() as db:
            return self.served(db, actor, via)

    def wake_reason(self, actor, previous, target):
        """Only useful peer updates or session endings wake a wait."""
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("ATTACH DATABASE ? AS bodydb", (str(self.bodies_path),))
        try:
            woken, ended = [], set()
            for row in db.execute("SELECT actor, kind, payload FROM events WHERE sequence > ? ORDER BY sequence",
                                  (previous or 0,)):
                payload = json.loads(row["payload"])
                if row["kind"] == "session":
                    if payload["state"] == "ended" and payload["actor"] != actor:
                        woken.append(dict(actor=payload["actor"], kind="ended"))
                elif row["actor"] != actor and row["kind"] in WAKE_KINDS:
                    woken.append(dict(actor=row["actor"], kind=row["kind"]))
            for _, _, payload in self.events(db, "session"):
                if payload["state"] == "ended":
                    ended.add(payload["actor"])
            peers = [a for a in self.actors if a != actor]
            if peers and all(a in ended for a in peers) and not any(w["kind"] != "ended" for w in woken):
                return dict(by="peer-ended", events=woken)
            if woken:
                return dict(by="update" if any(w["kind"] != "ended" for w in woken) else "peer-ended", events=woken)
            if target:
                for _, _, payload in self.events(db, "done", "body", "note"):
                    if payload.get("item") == target or payload.get("id") == target or payload.get("publication_id") == target:
                        return dict(by="already", events=[])
            return None
        finally:
            db.close()

    # ---- bodies -----------------------------------------------------------

    def with_originals(self, files):
        """Pair each validated file with the task's original version; done before any transaction opens."""
        stored = []
        for f in files:
            try:
                original = self.original(f["path"])
            except OriginalReadError as error:
                stored.append(dict(f, change="unknown", original_sha256=None, original=None,
                                   original_error=str(error)))
                continue
            change = "new" if original is None else "unchanged" if sha256(original) == f["sha256"] else "modified"
            stored.append(dict(f, change=change, original_sha256=None if original is None else sha256(original),
                               original=None if original is None else original.decode("utf-8", "replace")))
        return stored

    def register(self, db, actor, entry_id, stored):
        """Store the files behind an entry and log their names and sizes; content stays out of the log."""
        db.execute("INSERT INTO bodydb.bodies(id, time_ns, actor, files) VALUES (?, ?, ?, ?)",
                   (entry_id, time.monotonic_ns(), actor, json.dumps(stored, ensure_ascii=False)))
        self.append(db, actor, "body", dict(id=entry_id, files=metadata(stored)))

    def body(self, db, entry_id):
        entry_id = self.publication_id(db, entry_id)
        row = db.execute("SELECT actor, time_ns, files FROM bodydb.bodies WHERE id = ?", (entry_id,)).fetchone()
        if row is None:
            raise BoardError("no files are attached to an entry with this id")
        return row["actor"], row["time_ns"], json.loads(row["files"])

    @staticmethod
    def peer_versions(db, path, actor):
        """Every version of a path registered by agents other than `actor`: content hash to registration time.

        A copy equal to one of these came from an apply, so another peer version
        may replace it, unless it was registered later than the one being
        applied. The caller's own registered versions are its own work and are
        never overwritten.
        """
        versions = {}
        for row in db.execute("SELECT time_ns, files FROM bodydb.bodies WHERE actor != ?", (actor,)):
            for f in json.loads(row["files"]):
                if f["path"] == path:
                    versions[f["sha256"]] = max(versions.get(f["sha256"], 0), row["time_ns"])
        return versions

    @staticmethod
    def select(files, paths):
        if not paths:
            return files
        by_path = {f["path"]: f for f in files}
        missing = [p for p in paths if p not in by_path]
        if missing:
            raise BoardError(f"no attached file named {', '.join(missing)}")
        return [by_path[p] for p in paths]

    # ---- writes ---------------------------------------------------------

    def session_state(self, actor, state):
        """Record observed process lifecycle without changing work ownership."""
        if actor not in self.actors or state not in ("running", "ended"):
            raise BoardError("unknown agent or session state")
        with self.connect() as db:
            previous = "starting"
            for _, _, payload in self.events(db, "session"):
                if payload["actor"] == actor:
                    previous = payload["state"]
            if previous == state:
                return
            if previous == "ended":
                raise BoardError("an ended session cannot restart")
            payload = dict(actor=actor, state=state, recorded_at_ns=time.time_ns())
            self.append(db, SYSTEM, "session", payload)

    def status(self, actor, text):
        with self.connect() as db:
            self.append(db, actor, "status", dict(text=limited("status", text)))
            return dict(saved=True, view=self.served(db, actor, "status"))

    def write_item(self, db, actor, subject, detail):
        identity = uuid.uuid4().hex
        self.append(db, actor, "item", dict(id=identity, subject=subject, detail=detail))
        return identity

    def publish(self, actor, items):
        """Either worker can publish work; matching subjects retain all existing state."""
        if not isinstance(items, list) or not items:
            raise BoardError("publish needs a non-empty list of items")
        checked = []
        for item in items:
            if not isinstance(item, dict):
                raise BoardError("each item must have a subject and detail")
            checked.append((limited("item_subject", item.get("subject")), limited("item_detail", item.get("detail"))))
        with self.connect() as db:
            self.check_actor(actor)
            known = {normalized(i["subject"]): i for i in self.queue_state(db)}
            result = []
            for subject, detail in checked:
                existing = known.get(normalized(subject))
                created = existing is None
                if created:
                    existing = dict(id=self.write_item(db, actor, subject, detail), subject=subject, detail=detail, state="open")
                    known[normalized(subject)] = existing
                result.append(dict(id=existing["id"], state=existing["state"], created=created,
                                   **({"owner": existing["owner"]} if "owner" in existing else {})))
            return dict(items=result, view=self.served(db, actor, "publish"))

    def claim(self, actor, item_id):
        """Claim an item; the response carries the whole board either way, and a won claim sets the status."""
        with self.connect() as db:
            self.check_actor(actor)
            queue = self.queue_state(db)
            by_id = {i["id"]: i for i in queue}
            if item_id not in by_id:
                raise BoardError("unknown item")
            target = by_id[item_id]
            if target["state"] != "open":
                return dict(item=item_id, owner=target["owner"], state=target["state"], won=False,
                            view=self.served(db, actor, "claim"))
            held = [i for i in queue if i["state"] == "claimed" and i["owner"] == actor]
            if held:
                raise BoardError(f"you already hold an unfinished item: {held[0]['id']}")
            self.append(db, actor, "claim", dict(item=item_id))
            self.append(db, actor, "status", dict(text=target["subject"], auto=True))
            return dict(item=item_id, owner=actor, state="claimed", won=True, view=self.served(db, actor, "claim"))

    def reserve(self, actor, subject, detail):
        """Create/claim one concrete scope atomically.

        Normalized subjects identify the same work; different subjects may still
        overlap semantically. Retries return the existing ownership or completion.
        """
        subject = limited("item_subject", subject)
        detail = limited("item_detail", detail)
        with self.connect() as db:
            self.check_actor(actor)
            queue = self.queue_state(db)
            target = next((i for i in queue if normalized(i["subject"]) == normalized(subject)), None)
            if target is not None and target["state"] != "open":
                answer = dict(item=target["id"], owner=target["owner"], state=target["state"],
                              won=target["state"] == "claimed" and target["owner"] == actor,
                              created=False, view=self.served(db, actor, "reserve"))
                if target["state"] == "done":
                    # The incremental view may already have omitted this completion.
                    publication_id = target.get("publication_id", target["id"])
                    answer["publication_id"] = publication_id
                    answer["summary"] = target["summary"]
                return answer
            held = [i for i in queue if i["state"] == "claimed" and i["owner"] == actor]
            if held:
                raise BoardError(f"you already hold an unfinished item: {held[0]['id']}")
            created = target is None
            if created:
                target = dict(id=self.write_item(db, actor, subject, detail), subject=subject)
            self.append(db, actor, "claim", dict(item=target["id"]))
            self.append(db, actor, "status", dict(text=target["subject"], auto=True))
            return dict(item=target["id"], owner=actor, state="claimed", won=True, created=created,
                        view=self.served(db, actor, "reserve"))

    def done(self, actor, item_id, summary, files=None, request_id=None):
        """Publish a completion and its attached files atomically."""
        summary = limited("done", summary)
        checked = self.with_originals(check_files(files)) if files else []
        fingerprint = self.fingerprint(dict(kind="done", item=item_id, summary=summary, files=files))
        with self.connect() as db:
            self.check_actor(actor)
            previous = self.replayed(db, actor, request_id, fingerprint)
            if previous:
                return self.publication_response(db, actor, previous, "done")
            target = next((i for i in self.queue_state(db) if i["id"] == item_id), None)
            if target is None:
                raise BoardError("unknown item")
            if target["state"] != "claimed" or target["owner"] != actor:
                raise BoardError("only the agent holding this item may complete it")
            publication_id = uuid.uuid4().hex
            payload = dict(item=item_id, summary=summary, publication_id=publication_id)
            row = self.record_publication(db, actor, "done", publication_id, payload, request_id, fingerprint, item_id)
            self.append(db, actor, "done", payload)
            if checked:
                self.register(db, actor, publication_id, checked)
            self.append(db, actor, "status", dict(text="done: " + target["subject"], auto=True))
            return self.publication_response(db, actor, row, "done")

    def note(self, actor, note_type, text, scope, basis, files=None, request_id=None):
        if note_type not in NOTE_TYPES:
            raise BoardError("note type must be FACT or FAIL")
        payload = dict(type=note_type, text=limited("note_text", text),
                       scope=limited("note_scope", scope), basis=limited("note_basis", basis))
        checked = self.with_originals(check_files(files)) if files else []
        fingerprint = self.fingerprint(dict(kind="note", payload=payload, files=files))
        with self.connect() as db:
            self.check_actor(actor)
            previous = self.replayed(db, actor, request_id, fingerprint)
            if previous:
                return self.publication_response(db, actor, previous, "note")
            publication_id = uuid.uuid4().hex
            payload["id"] = publication_id
            row = self.record_publication(db, actor, "note", publication_id, payload, request_id, fingerprint)
            self.append(db, actor, "note", payload)
            if checked:
                self.register(db, actor, publication_id, checked)
            return self.publication_response(db, actor, row, "note")

    def stop(self):
        """Release waiting reads before the server shuts down."""
        self.stopping.set()

    # ---- immutable publications -----------------------------------------

    def check_actor(self, actor):
        if actor not in self.actors:
            raise BoardError("unknown agent")

    @staticmethod
    def fingerprint(value):
        return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())

    def publication_id(self, db, entry_id):
        row = db.execute("SELECT id FROM bodydb.publications WHERE id = ?", (entry_id,)).fetchone()
        if row:
            return row["id"]
        row = db.execute("SELECT id FROM bodydb.publications WHERE item_id = ? ORDER BY rowid DESC LIMIT 1", (entry_id,)).fetchone()
        return row["id"] if row else entry_id

    def replayed(self, db, actor, request_id, fingerprint):
        if request_id is None:
            return None
        limited("request_id", request_id)
        row = db.execute("SELECT * FROM bodydb.publications WHERE actor = ? AND request_id = ?", (actor, request_id)).fetchone()
        if row and row["fingerprint"] != fingerprint:
            raise BoardError("request_id was already used for different publication content")
        return row

    def record_publication(self, db, actor, kind, publication_id, payload, request_id, fingerprint, item_id=None):
        db.execute("""INSERT INTO bodydb.publications
            (id, actor, kind, item_id, payload, fingerprint, request_id) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                   (publication_id, actor, kind, item_id, json.dumps(payload, ensure_ascii=False), fingerprint, request_id))
        return db.execute("SELECT * FROM bodydb.publications WHERE id = ?", (publication_id,)).fetchone()

    def publication_response(self, db, actor, row, via):
        response = dict(publication_id=row["id"], view=self.served(db, actor, via))
        if row["kind"] == "done":
            response.update(item=row["item_id"], queue_empty=all(i["state"] == "done" for i in self.queue_state(db)))
        else:
            response["id"] = row["id"]
        return response

    # ---- reads ----------------------------------------------------------

    def read(self, actor, full=False, wait_seconds=0, target=None):
        """A view now, or a view as soon as a peer posts something (`wait_seconds` > 0).

        A waiting read polls the log without holding a write transaction and
        returns the incremental view the moment a wake reason exists, with
        `waited` and `woke_by` (update, peer-ended, already, timeout). A wait
        that times out serves nothing and leaves the read cursor where it was;
        it is recorded as a `wait` event, never as an empty read. Naming a
        target sets an automatic "waiting for" status, once per target, so the
        owner can see it.
        """
        if not wait_seconds:
            with self.connect() as db:
                return self.served(db, actor, "read", full=full)
        wait_seconds = min(float(wait_seconds), WAIT_CAP_SECONDS)
        with self.connect() as db:
            previous = self.last_read(db, actor)
            if target:
                text = f"waiting for {target}"
                if self.current_status(db, actor) != text:
                    self.append(db, actor, "status", dict(text=text, auto=True))
        started = time.monotonic()
        while True:
            woke = self.wake_reason(actor, previous, target)
            if woke or self.stopping.is_set() or time.monotonic() - started >= wait_seconds:
                break
            time.sleep(WAIT_POLL_SECONDS)
        waited = round(time.monotonic() - started, 3)
        with self.connect() as db:
            record = dict(target=target, requested=wait_seconds, waited=waited,
                          woke_by=woke["by"] if woke else "timeout", woken=woke["events"] if woke else [])
            self.append(db, actor, "wait", record)
            if not woke:
                current = self.view(db, actor)
                return dict(reader=actor, updates=False, full=False, waited=waited, woke_by="timeout", woken=[],
                            sessions=current["sessions"], sequence=current["sequence"],
                            queue=[], notes=[], status=[])
            result = self.served(db, actor, "wait", full=full)
            result.update(updates=True, waited=waited, woke_by=record["woke_by"], woken=record["woken"])
            return result

    def attached(self, actor, entry_id):
        """The names of the files behind an entry, as the read already shows them; logged, not a pull."""
        with self.connect() as db:
            author, _, files = self.body(db, entry_id)
            self.append(db, actor, "attached", dict(id=entry_id, files=[f["path"] for f in files]))
            return dict(id=entry_id, author=author, files=metadata(files))

    def expand(self, actor, entry_id, paths=None):
        """The files behind an entry as diffs against the original task files, or named files in full."""
        with self.connect() as db:
            author, _, files = self.body(db, entry_id)
            chosen = self.select(files, paths)
            rendered = []
            for f in chosen:
                if paths or f["original"] is None:
                    text = f["content"]
                else:
                    text = "".join(difflib.unified_diff(f["original"].splitlines(True), f["content"].splitlines(True),
                                                        "original/" + f["path"], author + "/" + f["path"]))
                rendered.append(dict(path=f["path"], bytes=f["bytes"], change=f["change"],
                                     shown="full" if paths or f["original"] is None else "diff", text=text,
                                     **({"original_error": f["original_error"]} if f.get("original_error") else {})))
            self.append(db, actor, "expand", dict(id=entry_id, files=[r["path"] for r in rendered],
                                                  shown="full" if paths else "diff"))
            return dict(id=entry_id, author=author, files=rendered)

    def apply(self, actor, entry_id, local, paths=None):
        """Decide, per attached file, whether the caller's copy may be overwritten; log the decision.

        `local` maps each attached path to the sha256 of the caller's current
        file, or None when absent. A copy that equals the original task file or
        a version a peer registered is overwritten (`write`), unless that peer
        version was registered later than the one being applied (`kept`: the
        caller already holds the newer one); a copy equal to the attached version
        is left alone (`unchanged`); anything else, including a version the
        caller registered itself, is the caller's own work (`conflict`) and
        makes the whole apply refused.
        """
        with self.connect() as db:
            author, registered_at, files = self.body(db, entry_id)
            chosen = self.select(files, paths)
            decisions, granted = {}, []
            for f in chosen:
                current = local.get(f["path"])
                peers = self.peer_versions(db, f["path"], actor)
                if current == f["sha256"]:
                    decisions[f["path"]] = "unchanged"
                elif current in peers and peers[current] > registered_at:
                    decisions[f["path"]] = "kept"
                elif current is None or (not f.get("original_error") and current == f["original_sha256"]) or current in peers:
                    decisions[f["path"]] = "write"
                    granted.append(dict(path=f["path"], content=f["content"], mode=f["mode"]))
                else:
                    decisions[f["path"]] = "conflict"
            applied = "conflict" not in decisions.values()
            self.append(db, actor, "apply", dict(id=entry_id, decisions=decisions, applied=applied,
                                                 confirmed=False))
            return dict(id=entry_id, author=author, applied=applied, decisions=decisions,
                        files=granted if applied else [])

    def apply_error(self, actor, entry_id, message):
        with self.connect() as db:
            self.append(db, actor, "apply_error", dict(id=entry_id, error=str(message)[:500]))

    def apply_result(self, actor, entry_id, written, success):
        with self.connect() as db:
            self.append(db, actor, "apply_result", dict(id=entry_id, written=written, success=success))
            return dict(saved=True, view=self.served(db, actor, "apply_result"))

    def export(self):
        with self.connect() as db:
            return [dict(sequence=row["sequence"], time_ns=row["time_ns"], actor=row["actor"],
                         kind=row["kind"], payload=json.loads(row["payload"]))
                    for row in db.execute("SELECT * FROM events ORDER BY sequence")]

    def export_bodies(self):
        with self.connect() as db:
            return [dict(id=row["id"], time_ns=row["time_ns"], actor=row["actor"], files=json.loads(row["files"]))
                    for row in db.execute("SELECT * FROM bodydb.bodies ORDER BY time_ns")]
