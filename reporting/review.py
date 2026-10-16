"""Capture submissions and export a review packet without assigning correctness."""
import json
import secrets
import subprocess


def capture(container, row, workdir):
    with (container.folder / "workspace.tar").open("wb") as archive:
        subprocess.run(["docker", "cp", f"{container.name}:{workdir.rstrip('/')}/.", "-"],
                       stdout=archive, stderr=subprocess.PIPE, check=True, timeout=60)
    if row.get("completed"):
        (container.folder / "final.txt").write_text(row.get("answer") or "")
    row["grading"] = {"status": "ungraded", "evaluator": "review"}
    row["correct"] = None


def export(folder, instruction, rubric, workers):
    packet = folder / "review-packet"
    packet.mkdir()
    (packet / "PROBLEM.txt").write_text(instruction)
    if rubric is not None:
        (packet / "rubric.txt").write_bytes(rubric)
    mapping = {}
    for worker in workers:
        if not worker.get("completed"):
            continue
        label = "proof-" + secrets.token_hex(8)
        (packet / (label + ".txt")).write_text(worker.get("answer") or "")
        mapping[label] = worker["actor"]
    (folder / "review-mapping-private.json").write_text(json.dumps(mapping, indent=2))
    (packet / "reviews.json").write_text(json.dumps(
        {label: {"verdict": "ungraded", "reviewer": None} for label in mapping}, indent=2))
