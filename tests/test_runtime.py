import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

from runtime import runner, delm_timing
from runtime.egress import destination, RestrictedNetwork
from runtime.containers import Container
from reporting.metrics import summarize
from tests.test_external_tasks import config
from benchmark.external_task import ExternalTask


class TimingAndMetricsTests(unittest.TestCase):
    def test_default_budget_does_not_wait_for_release(self):
        with patch.dict("os.environ", {"AGENT_TIMEOUT_SECONDS": "10", "DELM_TIMING": "per-worker"}):
            self.assertEqual(delm_timing.await_release(), 10)

    def test_common_deadline_uses_remaining_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "release.json"
            path.write_text(json.dumps({"deadline_ns": 9000000000}))
            with patch.dict("os.environ", {"AGENT_TIMEOUT_SECONDS": "10", "DELM_TIMING": "synchronized"}), patch.object(delm_timing, "Path", return_value=path), patch.object(delm_timing.time, "monotonic_ns", return_value=2000000000):
                self.assertEqual(delm_timing.await_release(), 7)
            path.write_text(json.dumps({"deadline_ns": 0}))
            with patch.dict("os.environ", {"AGENT_TIMEOUT_SECONDS": "10", "DELM_TIMING": "synchronized"}), patch.object(delm_timing, "Path", return_value=path):
                with self.assertRaises(RuntimeError):
                    delm_timing.await_release()

    def test_missing_release_is_bounded(self):
        with patch.dict("os.environ", {"AGENT_TIMEOUT_SECONDS": "10", "DELM_TIMING": "synchronized"}), patch.object(delm_timing, "Path") as path, patch.object(delm_timing.time, "monotonic", side_effect=[0, 121]):
            path.return_value.exists.return_value = False
            with self.assertRaisesRegex(RuntimeError, "did not release"):
                delm_timing.await_release()

    def test_metrics_preserve_legacy_origin_and_unknown_grades(self):
        record = dict(count=2, harness="codex", case="task", model="model", effort="xhigh",
                      startup_protocol="worker-planned", started_ns=1000000000, timeout_seconds=10,
                      workers=[dict(actor=f"agent-{i}", started_ns=1000000000,
                                    model_turn_start_seconds=1, agent_result_received_seconds=i+2,
                                    completed=True, process_returncode=0, correct=True, usage_agrees=True,
                                    native_usage=dict(session_id=str(i), input_tokens=10,
                                                      cached_input_tokens=5, output_tokens=2, total_tokens=12))
                               for i in (1, 2)])
        self.assertEqual(summarize(record)["last_worker_seconds"], 3)
        record.update(timing_policy="synchronized", release_ns=1500000000, evaluator="review")
        for worker in record["workers"]:worker["correct"] = None
        result = summarize(record)
        self.assertEqual(result["last_worker_seconds"], 3.5)
        self.assertIsNone(result["passing_workers"])
        self.assertNotIn("first_success_seconds", result)
        record["workers"][0]["agent_result_received_seconds"] = 20
        with self.assertRaises(ValueError):summarize(record)
        record["workers"][0]["error"] = "capacity"
        with self.assertRaises(ValueError):summarize(record)


class NetworkTests(unittest.TestCase):
    def test_proxy_allowlist(self):
        self.assertEqual(destination("chatgpt.com:443"), ("chatgpt.com", 443))
        for value in ("example.org:443", "chatgpt.com:80", "chatgpt.com.evil:443", "user@chatgpt.com:443"):
            with self.assertRaises(ValueError):destination(value)

    def test_container_network_is_opt_in_and_resource_limits_preserved(self):
        for network in (None, "restricted"):
            with self.subTest(network=network), tempfile.TemporaryDirectory() as tmp:
                limits = {"environment": {"cpus": 2, "memory_mb": 4096}}
                if network:limits["network_name"] = network
                with patch("runtime.containers.subprocess.run") as docker:
                    Container("image", Path(tmp), limits, "/workspace", {}, [])
                command = docker.call_args.args[0]
                self.assertEqual("--network" in command, bool(network))
                self.assertEqual(command[command.index("--cpus") + 1], "2")
                self.assertEqual(command[command.index("--memory") + 1], "4096m")


class SharedRunnerTests(unittest.TestCase):
    def run_trial(self, external=True, capture_error=False, network_error=False, worker_error=False):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            task = root / "task"
            task.mkdir()
            path = config(task, files={}) if external else None
            (task / "instruction.md").write_text("coding task")
            limits = {"agent": {"timeout_sec": 180}, "environment": {"cpus": 2, "memory_mb": 4096, "storage_mb": 1}}
            image = {"Id": "sha256:" + "a" * 64, "Config": {"WorkingDir": "/workspace", "Env": ["PATH=/usr/bin"]}}
            def mock(target, **kwargs):return stack.enter_context(patch(target, **kwargs))
            stack.enter_context(patch("sys.stdout", new=io.StringIO()))
            mock("runtime.runner.ROOT", new=root)
            mock("runtime.runner.source_identity", return_value=("commit", {}))
            mock("runtime.runner.source_files", return_value=[])
            mock("runtime.runner.preflight", return_value=({}, limits, {}, image, image if not external else None))
            mock("runtime.runner.create_original", return_value=SimpleNamespace(name="original", fetch=lambda path: None, remove=lambda: None))
            server = MagicMock(server_port=12345)
            mock("runtime.runner.ThreadingHTTPServer", return_value=server)
            mock("runtime.runner.board_address", return_value="127.0.0.1")
            restricted = MagicMock()
            restricted.start.return_value = "127.0.0.1"
            restricted.name = "restricted"
            restricted.environment.return_value = {}
            if network_error:restricted.close.side_effect=RuntimeError("network cleanup failed")
            mock("runtime.runner.RestrictedNetwork", return_value=restricted)
            mock("runtime.runner.runtime.worker_mounts", return_value=[])
            processes = []
            def container(image, folder, limits, workdir, environment, mounts, **kwargs):
                folder.mkdir(parents=True)
                (folder / "context-command-errors.jsonl").touch()
                result = dict(completed=not worker_error, started_ns=1000000000,
                              model_turn_start_seconds=1, answer="proof", tokens={"input": 10, "output": 2})
                if worker_error:result["error"]="provider failure"
                (folder / "result.json").write_text(json.dumps(result))
                process = SimpleNamespace(poll=lambda: 0, wait=lambda **kwargs: 0, returncode=0)
                processes.append(process)
                return SimpleNamespace(name=folder.name, folder=folder, compose=None,
                                       preflight=lambda *args: None, start=lambda _: (process, io.StringIO()))
            create = mock("runtime.runner.create_container", side_effect=container)
            def watch(process, log, lifecycle, started, context, actor):
                lifecycle.update(agent_ready_received_seconds=0.1, agent_model_start_received_seconds=1,
                                 agent_result_received_seconds=2, finished_seconds=3)
            mock("runtime.runner.watch", side_effect=watch)
            class Thread:
                def __init__(self, target, args=(), **kwargs):self.target,self.args=target,args
                def start(self):self.target(*self.args)
                def join(self, **kwargs):pass
                def is_alive(self):return False
            mock("runtime.runner.threading.Thread", new=Thread)
            mock("runtime.runner.time.monotonic_ns", return_value=1000000000)
            def collect(harness, container, row, model):
                row.update(usage_agrees=True, native_usage=dict(session_id=container.name,
                           input_tokens=10, cached_input_tokens=5, output_tokens=2, total_tokens=12))
            mock("runtime.runner.runtime.collect_record", side_effect=collect)
            capture = mock("runtime.runner.review.capture", side_effect=RuntimeError("archive failed") if capture_error else lambda c,r,w:r.update(correct=None, grading={"status":"ungraded"}))
            official = mock("runtime.runner.capture_and_grade", side_effect=lambda task,cs,rs,*args:[r.update(correct=True, grading={"reward":1}) for r in rs])
            stop = mock("runtime.runner.stop_checked")
            remove = mock("runtime.runner.remove_container")
            result = runner.run.__wrapped__(task, "model", "xhigh", "series", 1, Path("/auth"), task_config=path)
            saved = json.loads((root / ".local/runs/series" / result["case"] / "repeat-1/run.json").read_text())
            self.assertEqual(saved["status"], result["status"])
            self.assertEqual(create.call_count, 2)
            self.assertEqual(remove.call_count, 2)
            self.assertGreaterEqual(stop.call_count, 2)
            self.assertEqual(official.call_count, int(not external))
            if external:
                self.assertEqual(capture.call_count, 2)
                restricted.close.assert_called_once()
                self.assertEqual(result["timing_policy"], "synchronized")
            else:
                capture.assert_not_called()
                restricted.start.assert_not_called()
                self.assertNotIn("release_ns", result)
            return result

    def test_external_task_uses_shared_lifecycle_and_stays_ungraded(self):
        result = self.run_trial()
        self.assertEqual(result["status"], "finished")
        self.assertIsNone(result["summary"]["passing_workers"])

    def test_terminal_task_keeps_official_grading_and_timing(self):
        result = self.run_trial(external=False)
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["summary"]["passing_workers"], 2)
        self.assertEqual(result["timing_policy"], "per-worker")

    def test_capture_failure_preserves_results_and_cleans_up(self):
        result = self.run_trial(capture_error=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertNotIn("summary", result)
        self.assertEqual(len(result["workers"]), 2)

    def test_cleanup_failure_invalidates_summary(self):
        result = self.run_trial(network_error=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertNotIn("summary", result)
        self.assertTrue(result["cleanup_errors"])

    def test_worker_failure_is_not_success_or_zero_usage(self):
        result = self.run_trial(worker_error=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertNotIn("summary", result)


class PreflightTests(unittest.TestCase):
    def test_official_image_checks_remain_and_external_image_is_pinned(self):
        for external_mode in (False, True):
            with self.subTest(external=external_mode), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                task = Path(tmp)
                external = ExternalTask(config(task)) if external_mode else None
                limits = {"agent": {"timeout_sec": 180},
                          "environment": {"cpus": 2, "memory_mb": 4096, "storage_mb": 1},
                          "verifier": {"environment_mode": "separate"}}
                image = {"Id": "sha256:" + "a" * 64, "Config": {"WorkingDir": "/workspace"}}
                stack.enter_context(patch("runtime.runner.check_source", return_value={}))
                stack.enter_context(patch("runtime.runner.settings", return_value=limits))
                stack.enter_context(patch("runtime.runner.runtime.validate_key_file"))
                stack.enter_context(patch("runtime.runner.subprocess.check_output", return_value=""))
                stack.enter_context(patch("runtime.runner.os.sched_getaffinity", return_value=set(range(8)), create=True))
                stack.enter_context(patch("runtime.runner.shutil.disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)))
                def proc(path):
                    if str(path) == "/proc/meminfo":
                        return SimpleNamespace(exists=lambda: True, read_text=lambda: "MemAvailable: 32000000 kB")
                    if str(path) == "/proc/cpuinfo":
                        return SimpleNamespace(read_text=lambda: "model name: test CPU")
                    return Path(path)
                stack.enter_context(patch("runtime.runner.Path", side_effect=proc))
                inspect = stack.enter_context(patch("runtime.runner.inspect_image", return_value=image))
                source = stack.enter_context(patch("runtime.runner.validate_source"))
                overlay = stack.enter_context(patch("runtime.runner.runtime.validate_overlay"))
                result = runner.preflight(task, Path("/auth"), 0, 0, external=external)
                self.assertEqual(result[2]["task_and_service_cpus"], 4)
                self.assertEqual(inspect.call_count, 1 if external_mode else 3)
                self.assertEqual(source.call_count, 0 if external_mode else 3)
                self.assertEqual(overlay.call_count, int(not external_mode))
                if external_mode:
                    self.assertEqual(inspect.call_args.args[0], external.config["image"])
                    self.assertIsNone(result[-1])
                    image["Config"].pop("WorkingDir")
                    with self.assertRaisesRegex(ValueError, "non-root working directory"):
                        runner.preflight(task, Path("/auth"), 0, 0, external=external)

    def test_provider_preflight_code_uses_selected_network_policy(self):
        from runtime.runtime_support import WorkerContainer
        for restricted in (False, True):
            with self.subTest(restricted=restricted), tempfile.TemporaryDirectory() as tmp:
                worker = WorkerContainer.__new__(WorkerContainer)
                worker.name, worker.folder, worker.workdir = "worker", Path(tmp), "/workspace"
                worker.harness, worker.authentication = "codex", "chatgpt"
                worker.restricted_network = restricted
                with patch("runtime.runtime_support.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout='{}', stderr='')) as call:
                    worker.preflight(1, "http://gateway:1234")
                code = call.call_args.args[0][-1]
                compile(code, "preflight", "exec")
                self.assertEqual("set_tunnel" in code, restricted)
                self.assertEqual("connect_ex" in code, restricted)
                self.assertIn("worker.preflight", code)
