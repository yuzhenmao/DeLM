"""Protect local and cross-author edits for both supported worker counts."""

import json

from board.shared_context import SharedContext


class ProtectedBoard(SharedContext):
    """A later publication from another author is not proof of a merged file."""

    def apply(self, actor, entry_id, local, paths=None):
        with self.connect() as db:
            author, registered_at, files = self.body(db, entry_id)
            chosen = self.select(files, paths)
            history = {}
            for row in db.execute("SELECT id, actor, time_ns, files FROM bodydb.bodies"):
                for f in json.loads(row["files"]):
                    history.setdefault((f["path"], f["sha256"]), []).append(
                        dict(id=row["id"], author=row["actor"], time_ns=row["time_ns"]))
            decisions, granted, conflicts = {}, [], {}
            for f in chosen:
                path = f["path"]
                current = local.get(path)
                versions = history.get((path, current), [])
                if current == f["sha256"]:
                    decision = "unchanged"
                elif current is None or (not f.get("original_error") and current == f["original_sha256"]):
                    decision = "write"
                elif versions and {v["author"] for v in versions} == {author} and author != actor:
                    decision = "kept" if max(v["time_ns"] for v in versions) > registered_at else "write"
                else:
                    decision = "conflict"
                    conflicts[path] = dict(
                        reason="Local edits or another author's version require local reconciliation",
                        incoming_publication=entry_id, incoming_author=author,
                        current_publications=versions)
                decisions[path] = decision
                if decision == "write":
                    granted.append(dict(path=path, content=f["content"], mode=f["mode"]))
            applied = not conflicts
            self.append(db, actor, "apply", dict(id=entry_id, decisions=decisions,
                                                 applied=applied, confirmed=False,
                                                 conflicts=conflicts))
            return dict(id=entry_id, author=author, applied=applied, decisions=decisions,
                        conflicts=conflicts, files=granted if applied else [])
