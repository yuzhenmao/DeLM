import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import delm
from benchmark.external_task import ExternalTask
from reporting import review


def config(folder, **overrides):
    (folder / "problem.txt").write_text("Prove 1 + 1 = 2.")
    (folder / "prompt.txt").write_text("{instruction} Worker {worker}; peers {roster}; count {count}; {workspace}")
    (folder / "rubric.txt").write_text("CONTROLLER_ONLY")
    value = dict(name="arithmetic", image="sha256:" + "a" * 64,
                 prompt="prompt.txt", rubric="rubric.txt",
                 files={"PROBLEM.txt": "problem.txt"})
    value.update(overrides)
    path = folder / "task.json"
    path.write_text(json.dumps(value))
    return path


class ExternalTaskTests(unittest.TestCase):
    def test_snapshot_is_allowlisted_and_stable_after_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = config(folder)
            (folder / "auth.json").write_text("SECRET")
            task = ExternalTask(path)
            identity = task.identity()
            (folder / "problem.txt").write_text("changed")
            task.snapshot(folder / "snapshot")
            self.assertEqual(task.identity(), identity)
            self.assertEqual((folder / "snapshot/workspace/PROBLEM.txt").read_text(), "Prove 1 + 1 = 2.")
            self.assertFalse((folder / "snapshot/auth.json").exists())
            self.assertNotIn(b"CONTROLLER_ONLY", b"".join(task.initial_files.values()))
            self.assertNotIn("CONTROLLER_ONLY", task.render("agent-1", ["agent-1", "agent-2"], "/workspace"))
            self.assertNotEqual(identity, ExternalTask(path).identity())

    def test_rejects_invalid_configuration_before_execution(self):
        for updates in ({"image": "mutable:latest"}, {"cpus": True}, {"memory_mb": 0},
                        {"timeout_seconds": 0.5}, {"network": "typo"}, {"evaluator": "lean"},
                        {"timing": "typo"}, {"name": "../escape"}, {"unknown": 1},
                        {"files": {"../escape": "problem.txt"}},
                        {"files": {"/absolute": "problem.txt"}},
                        {"files": {"a": "problem.txt", "a/b": "problem.txt"}},
                        {"instruction": "../outside.txt"}):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    ExternalTask(config(Path(tmp), **updates))

    def test_rejects_symlink_and_bad_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = config(folder)
            (folder / "problem.txt").unlink()
            (folder / "problem.txt").symlink_to(folder / "rubric.txt")
            with self.assertRaises(ValueError):
                ExternalTask(path)
            path = config(folder, instruction="rubric.txt")
            (folder / "prompt.txt").write_text("{unknown}")
            with self.assertRaises((KeyError, ValueError)):
                ExternalTask(path)

    def test_seed_only_copies_explicit_workspace_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            task = ExternalTask(config(folder))
            task.snapshot(folder / "snapshot")
            with patch("benchmark.external_task.subprocess.run") as call:
                task.seed("container", folder / "snapshot", "/workspace")
                self.assertEqual(call.call_args.args[0][-2:], [str(folder / "snapshot/workspace") + "/.", "container:/workspace/"])

    def test_launcher_uses_same_runner_and_preserves_existing_defaults(self):
        args = delm.parser().parse_args(["existing-task", "--series", "test", "--api-key-file", "/missing/key"])
        plan = delm.build_plan(args)
        self.assertEqual((plan.n, plan.harness, plan.task_config), (2, "codex", None))
        with tempfile.TemporaryDirectory() as tmp:
            path = config(Path(tmp))
            args = delm.parser().parse_args(["--task-config", str(path), "--series", "test", "--api-key-file", "/missing/key"])
            plan = delm.build_plan(args)
            self.assertEqual(plan.task, "arithmetic")
            with patch("delm.sys.version_info", (3, 13)), patch("runtime.runner.run", return_value={"status": "finished"}) as run:
                self.assertEqual(delm.execute(plan), 0)
                self.assertEqual(run.call_args.kwargs["task_config"], str(path.resolve()))
            args.task = "extra"
            with self.assertRaises(ValueError):
                delm.build_plan(args)

    def test_review_exports_preserve_unknown_correctness_and_identity_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            workers = [{"actor": "agent-1", "completed": True, "answer": "proof"},
                       {"actor": "agent-2", "completed": False}]
            review.export(folder, "problem", b"rubric", workers)
            mapping = json.loads((folder / "review-mapping-private.json").read_text())
            self.assertEqual(list(mapping.values()), ["agent-1"])
            label = next(iter(mapping))
            self.assertEqual((folder / "review-packet" / (label + ".txt")).read_text(), "proof")
            self.assertEqual(json.loads((folder / "review-packet/reviews.json").read_text())[label]["verdict"], "ungraded")
            self.assertFalse((folder / "review-packet/review-mapping-private.json").exists())

    def test_capture_failure_propagates_and_never_sets_a_grade(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            container = SimpleNamespace(name="worker", folder=Path(tmp))
            row = {"completed": True, "answer": "proof"}
            with patch("reporting.review.subprocess.run", side_effect=subprocess.TimeoutExpired("cp", 60)):
                with self.assertRaises(subprocess.TimeoutExpired):
                    review.capture(container, row, "/workspace")
            self.assertNotIn("correct", row)
