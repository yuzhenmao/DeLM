# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

"""SWE-bench executor - based on official swebench harness implementation."""
import asyncio
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from base.engine.logs import logger
from benchmark.swebench.data_loader import SWEBenchInstance
from benchmark.swebench import swebench_grader

# ============================================================================
# Constants from official swebench (swebench/harness/constants.py)
# ============================================================================

START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

NON_TEST_EXTS = [
    ".json", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico",
    ".txt", ".md", ".rst", ".csv", ".tsv", ".xml", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".conf", ".lock", ".log",
]

# Repository-specific test commands (simplified from MAP_REPO_VERSION_TO_SPECS)
REPO_TEST_CMDS = {
    "astropy/astropy": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "django/django": "./tests/runtests.py --verbosity 2 {tests}",
    "matplotlib/matplotlib": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "pallets/flask": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "psf/requests": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "pylint-dev/pylint": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "pytest-dev/pytest": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "scikit-learn/scikit-learn": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "sphinx-doc/sphinx": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "sympy/sympy": "bin/test -C --verbose {tests}",
    "pydata/xarray": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "mwaskom/seaborn": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
}

DEFAULT_TEST_CMD = "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider"


# ============================================================================
# Utility functions from official swebench
# ============================================================================

def _iter_patch_file_blocks(patch: str):
    """Yield (source, target, diff_block) entries from a unified patch."""
    matches = list(re.finditer(r"^diff --git a/(.*?) b/(.*?)$", patch or "", re.M))
    for i, match in enumerate(matches):
        block_start = match.start()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(patch)
        yield match.group(1), match.group(2), patch[block_start:block_end]


def get_modified_files(patch: str) -> List[str]:
    """Extract modified files from a patch, excluding newly-created files."""
    return [
        source
        for source, _target, block in _iter_patch_file_blocks(patch)
        if not re.search(r"^--- /dev/null\s*$", block, re.M)
    ]


def get_new_files(patch: str) -> List[str]:
    """Extract newly-created files from a patch."""
    return [
        target
        for _source, target, block in _iter_patch_file_blocks(patch)
        if re.search(r"^--- /dev/null\s*$", block, re.M)
    ]


def get_reset_test_commands(base_commit: str, test_patch: str) -> List[str]:
    """Current SWE-bench reset semantics for gold test files."""
    modified_files = get_modified_files(test_patch)
    new_files = get_new_files(test_patch)
    commands: List[str] = []
    if modified_files:
        commands.append(f"git checkout {base_commit} {' '.join(modified_files)}")
    if new_files:
        commands.append(f"rm -f {' '.join(new_files)}")
    return commands


def get_test_directives(repo: str, test_patch: str) -> List[str]:
    """
    Get test directives from the test_patch of a task instance.
    Based on swebench/harness/test_spec/python.py:get_test_directives
    """
    diff_pat = r"diff --git a/.* b/(.*)"
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]
    
    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots
    if repo == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed
    
    return directives


def make_eval_script(
    repo: str,
    base_commit: str,
    test_patch: str,
    repo_directory: str = "/testbed",
    env_name: str = "testbed",
) -> str:
    """
    Generate evaluation script based on official swebench implementation.
    Based on swebench/harness/test_spec/python.py:make_eval_script_list_py
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    
    # Get test directives
    test_directives = get_test_directives(repo, test_patch) if test_patch else []
    directives_str = " ".join(test_directives) if test_directives else ""
    
    # Get test command for repo
    test_cmd_template = REPO_TEST_CMDS.get(repo, DEFAULT_TEST_CMD)
    test_cmd = test_cmd_template.format(tests=directives_str)
    
    # Reset test files command. Current SWE-bench splits modified files from
    # newly-created files: modified files are checked out from base, while new
    # files are removed. A checkout with no path args would reset the whole repo.
    reset_tests_commands = get_reset_test_commands(base_commit, test_patch) if test_patch else []
    reset_tests_block = "\n".join(reset_tests_commands) if reset_tests_commands else ":"
    
    # Apply test patch command
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        if test_patch else ":"
    )
    
    # Build eval script following official pattern
    eval_commands = [
        "#!/bin/bash",
        "set -uxo pipefail",  # Don't use -e to allow tests to fail
        "",
        "# Activate conda environment",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
        "",
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
        "",
        "# Informational output",
        "git status",
        "git show --stat",
        f"git -c core.fileMode=false diff {base_commit}",
        "",
        "# Re-activate environment",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        "",
        "# Reset test files to base commit state",
        reset_tests_block,
        "",
        "# Apply test patch",
        apply_test_patch_command,
        "",
        "# Run tests",
        f": '{START_TEST_OUTPUT}'",
        test_cmd,
        f": '{END_TEST_OUTPUT}'",
        "",
        "# Revert test files",
        reset_tests_block,
    ]
    
    return "\n".join(eval_commands) + "\n"


class SWEBenchExecutor:
    """Executes SWE-bench tasks using Docker containers."""

    def __init__(
        self,
        instance: SWEBenchInstance,
        logs_dir: Path,
        timeout: int = 1800,
        env_init: Optional[Dict[str, str]] = None,
        name_suffix: str = "",
        grading_timeout: Optional[int] = None,
    ):
        self.instance = instance
        self.logs_dir = logs_dir
        self.timeout = timeout
        self.grading_timeout = grading_timeout if grading_timeout is not None else min(timeout, 600)
        self.env_init = env_init or {}
        self.name_suffix = str(name_suffix or "")

        self.container_id: Optional[str] = None
        self._temp_dir: Optional[Path] = None
        self._repo_path: Optional[str] = None  # Linux path in container, use str not Path

        self.grading_deadline_perf: Optional[float] = None

        # Create logs directory
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    async def start_container(self):
        """Start Docker container for the SWE-bench instance."""
        # Create temporary directory for workspace
        self._temp_dir = Path(tempfile.mkdtemp(prefix=f"swebench_{self.instance.instance_id}{self.name_suffix}_"))
        
        # Match mini-swe-agent's SWE-bench image naming exactly.
        id_docker_compatible = self.instance.instance_id.replace("__", "_1776_")
        image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        
        logger.info(f"Starting container for {self.instance.instance_id}")
        logger.info(f"Image: {image_name}")
        
        try:
            # Check if image exists locally
            check_result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "images", "-q", image_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if not check_result.stdout.strip():
                # Pull image
                logger.info(f"Pulling image: {image_name}")
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "pull", image_name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to pull image: {result.stderr}")
            
            container_name = f"swebench_{self.instance.instance_id}{self.name_suffix}"
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=30,
            )
            
            # Run container
            env_args = []
            for key, value in self.env_init.items():
                env_args.extend(["-e", f"{key}={value}"])
            
            cmd = [
                "docker", "run", "-d",
                "--name", container_name,
                "-v", f"{self._temp_dir}:/workspace",
                *env_args,
                image_name,
                "tail", "-f", "/dev/null",  # Keep container running
            ]
            
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start container: {result.stderr}")
            
            self.container_id = result.stdout.strip()
            logger.info(f"Container started: {self.container_id[:12]}")
            
            # Setup repository in container
            await self._setup_repo()
            
        except Exception as e:
            await self.cleanup()
            raise RuntimeError(f"Failed to start container: {e}") from e

    async def _setup_repo(self):
        """Use the prepared SWE-bench repository already baked into the image."""
        if not self.container_id:
            raise RuntimeError("Container not started")
        
        # SWE-bench official images always place the repository at /testbed
        self._repo_path = "/testbed"

        logger.info("Using prepared SWE-bench image repository at /testbed")
        output, exit_code = await self.execute_command(
            f"cd {self._repo_path} "
            f"&& git config --global --add safe.directory {self._repo_path} "
            "&& git rev-parse --verify HEAD "
            "&& git status --short"
        )
        if exit_code != 0:
            logger.warning(f"Prepared repository check failed: {output}")

    async def execute_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
        idle_timeout: Optional[int] = None,
    ) -> Tuple[str, int]:
        """Execute command in container.

        Uses stdin to pass command to avoid Windows command line length limit (~8191 chars).
        This allows executing commands with large content (e.g., base64 encoded files).

        idle_timeout: kill the command if it produces no output for this many
        seconds (catches hung agent commands blocked on network/stdin). Defaults
        to 180s. Long, legitimately-quiet runs (e.g. the official grade) should
        pass a larger value so they aren't killed by silence alone.
        """
        if not self.container_id:
            raise RuntimeError("Container not started")

        exec_timeout = timeout if timeout is not None else self.timeout
        
        try:
            cmd = ["docker", "exec", "-i"]
            if workdir:
                cmd.extend(["-w", workdir])
            cmd.extend([self.container_id, "bash"])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            full_command = (
                "source /opt/miniconda3/bin/activate 2>/dev/null\n"
                "conda activate testbed 2>/dev/null\n"
                f"{command}"
            )

            import shlex as _shlex, time as _time
            hard = max(1, int(exec_timeout))
            idle_timeout = int(
                idle_timeout if idle_timeout is not None
                else (getattr(self, "idle_timeout", 0) or 180)
            )
            _D = "__ACI_CMD_EOF_7Q__"
            wrapped = (
                f"cat > /tmp/.aci_cmd.sh <<'{_D}'\n{full_command}\n{_D}\n"
                f"exec timeout --kill-after=10 {hard} bash /tmp/.aci_cmd.sh\n"
            )
            proc.stdin.write(wrapped.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            chunks: list = []
            start = _time.monotonic()
            killed = None
            while True:
                abs_left = hard + 20 - (_time.monotonic() - start)
                if abs_left <= 0:
                    killed = f"hard cap {hard}s"
                    break
                try:
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(65536), timeout=min(idle_timeout, abs_left)
                    )
                except asyncio.TimeoutError:
                    killed = (
                        f"hard cap {hard}s"
                        if (_time.monotonic() - start) >= hard
                        else f"idle {idle_timeout}s (no output — likely blocked, e.g. network/stdin)"
                    )
                    break
                if not chunk:
                    break  # EOF: the command finished and closed the pipe
                chunks.append(chunk)

            output = b"".join(chunks).decode("utf-8", errors="replace")
            if killed is not None:
                try:
                    proc.kill()  # stop waiting on the docker-exec client
                except Exception:
                    pass
                try:
                    if self.container_id:
                        p2 = await asyncio.create_subprocess_exec(
                            "docker", "exec", self.container_id, "bash", "-c",
                            "pkill -9 -f .aci_cmd.sh; pkill -9 -f 'timeout --kill-after'; "
                            "pkill -9 -f eval.sh; pkill -9 -f pytest; pkill -9 -f runtests; "
                            "pkill -9 -f sphinx-build; true",
                            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                        )
                        await asyncio.wait_for(p2.communicate(), timeout=10)
                except Exception:
                    pass
                return output + f"\n[Command terminated by watchdog: {killed}]", -1

            await proc.wait()
            return output, proc.returncode or 0

        except Exception as e:
            return f"Error executing command: {e}", -1

    async def run_tests(self) -> Tuple[float, Dict[str, Any]]:
        """Run tests and return reward and details.
        
        Based on official swebench harness implementation.
        """
        if not self.container_id:
            raise RuntimeError("Container not started")

        model_patch = ""
        try:
            cleanup_out, cleanup_code = await self.execute_command(
                f"cd {self._repo_path} && "
                "git -c core.fileMode=false clean -fd -- "
                "_build .doctrees __pycache__ .pytest_cache .mypy_cache "
                ".ruff_cache .tox .nox build dist htmlcov "
                "':(glob)**/_build' ':(glob)**/.doctrees' "
                "':(glob)**/__pycache__' ':(glob)**/.pytest_cache' "
                "':(glob)**/.mypy_cache' ':(glob)**/.ruff_cache' "
                "':(glob)**/.tox' ':(glob)**/.nox' "
                "':(glob)**/build' ':(glob)**/dist' ':(glob)**/htmlcov'",
                timeout=30,
            )
            if cleanup_code not in (0, None):
                logger.warning(
                    f"[SWEBenchExecutor] generated-artifact cleanup failed: {cleanup_out}"
                )
            elif str(cleanup_out or "").strip():
                logger.info(
                    "[SWEBenchExecutor] cleaned generated artifacts before "
                    f"model_patch capture:\n{cleanup_out.strip()}"
                )
            mp_out, _mp_rc = await self.execute_command(
                f"cd {self._repo_path} "
                f"&& git -c core.fileMode=false add -A "
                f"&& git -c core.fileMode=false diff --cached HEAD"
                f"; git -c core.fileMode=false reset -q"
            )
            model_patch = mp_out if isinstance(mp_out, str) else ""
            (self.logs_dir / "model_patch.diff").write_text(model_patch, encoding="utf-8")
            logger.info(
                f"[SWEBenchExecutor] Saved model_patch ({len(model_patch)} chars) -> "
                f"{self.logs_dir / 'model_patch.diff'}"
            )
        except Exception as e:
            logger.warning(f"[SWEBenchExecutor] model_patch capture failed: {e}")

        instance_dict = {
            "instance_id": self.instance.instance_id,
            "repo": self.instance.repo,
            "version": self.instance.version,
            "base_commit": self.instance.base_commit,
            "test_patch": self.instance.test_patch or "",
            "patch": self.instance.patch or "",
            "problem_statement": self.instance.problem_statement or "",
            "hints_text": self.instance.hints_text or "",
            "environment_setup_commit": self.instance.environment_setup_commit or self.instance.base_commit,
            "FAIL_TO_PASS": self.instance.FAIL_TO_PASS or [],
            "PASS_TO_PASS": self.instance.PASS_TO_PASS or [],
        }
        eval_script = swebench_grader.official_eval_script(
            instance_dict, repo_directory=self._repo_path, env_name="testbed"
        )
        if eval_script:
            logger.info("[SWEBenchExecutor] Using OFFICIAL swebench eval script")
        else:
            logger.warning(
                "[SWEBenchExecutor] Official eval script unavailable "
                f"(repo={self.instance.repo} version={self.instance.version}); "
                "falling back to local make_eval_script"
            )
            eval_script = make_eval_script(
                repo=self.instance.repo,
                base_commit=self.instance.base_commit,
                test_patch=self.instance.test_patch or "",
                repo_directory=self._repo_path,
                env_name="testbed",
            )

        # Save eval script to log for debugging
        eval_script_log = self.logs_dir / "eval.sh"
        with eval_script_log.open("w", encoding="utf-8") as f:
            f.write(eval_script)
        
        # Write eval script to container and execute
        await self.execute_command(
            f"cat > /eval.sh << 'EOF_EVAL_SCRIPT'\n{eval_script}\nEOF_EVAL_SCRIPT"
        )
        await self.execute_command("chmod +x /eval.sh")
        
        eff_grading_timeout = self.grading_timeout
        if self.grading_deadline_perf is not None:
            import time as _time_grade
            remaining = self.grading_deadline_perf - _time_grade.perf_counter()
            eff_grading_timeout = int(max(15, min(self.grading_timeout, remaining)))
            if remaining < self.grading_timeout:
                logger.info(
                    f"[SWEBenchExecutor] grading window bounded by task deadline: "
                    f"{eff_grading_timeout}s (grading_timeout={self.grading_timeout}s, "
                    f"remaining={remaining:.0f}s)"
                )
        test_output, exit_code = await self.execute_command(
            "/bin/bash /eval.sh",
            timeout=eff_grading_timeout,
            idle_timeout=eff_grading_timeout,
        )
        if exit_code == -1:
            logger.warning(
                f"[SWEBenchExecutor] grading terminated by watchdog (cap "
                f"{eff_grading_timeout}s; likely a non-terminating patch or out of "
                "task budget); scoring as unresolved"
            )
        
        # Save test output to log
        test_output_log = self.logs_dir / "test_output.txt"
        with test_output_log.open("w", encoding="utf-8") as f:
            f.write(test_output)
        
        resolved, test_results = swebench_grader.grade(
            repo=self.instance.repo,
            test_output=test_output,
            fail_to_pass=self.instance.FAIL_TO_PASS,
            pass_to_pass=self.instance.PASS_TO_PASS,
        )

        # Build results dict (preserve the historical shape + keys)
        results = {
            "fail_to_pass": {
                "passed": test_results["FAIL_TO_PASS"]["success"],
                "failed": test_results["FAIL_TO_PASS"]["failure"],
            },
            "pass_to_pass": {
                "passed": test_results["PASS_TO_PASS"]["success"],
                "failed": test_results["PASS_TO_PASS"]["failure"],
            },
        }

        fail_to_pass_total = len(self.instance.FAIL_TO_PASS) if self.instance.FAIL_TO_PASS else 0
        fail_to_pass_success = len(results["fail_to_pass"]["passed"])
        pass_to_pass_total = len(self.instance.PASS_TO_PASS) if self.instance.PASS_TO_PASS else 0
        pass_to_pass_success = len(results["pass_to_pass"]["passed"])

        reward = 1.0 if resolved else 0.0
        
        results["reward"] = reward
        results["model_patch"] = model_patch
        results["summary"] = {
            "fail_to_pass": f"{fail_to_pass_success}/{fail_to_pass_total}",
            "pass_to_pass": f"{pass_to_pass_success}/{pass_to_pass_total}",
        }
        
        # Save test results to log
        test_log = self.logs_dir / "test_results.log"
        with test_log.open("w", encoding="utf-8") as f:
            f.write(f"Instance: {self.instance.instance_id}\n")
            f.write(f"Resolved: {resolved}\n")
            f.write(f"Reward: {reward}\n")
            f.write(f"FAIL_TO_PASS: {fail_to_pass_success}/{fail_to_pass_total}\n")
            f.write(f"PASS_TO_PASS: {pass_to_pass_success}/{pass_to_pass_total}\n")
            f.write(f"\nDetailed results:\n")
            f.write(f"F2P passed: {results['fail_to_pass']['passed']}\n")
            f.write(f"F2P failed: {results['fail_to_pass']['failed']}\n")
            f.write(f"P2P passed: {results['pass_to_pass']['passed']}\n")
            f.write(f"P2P failed: {results['pass_to_pass']['failed']}\n")
        
        return reward, results

    async def get_file_content(self, file_path: str) -> Tuple[str, int]:
        """Read file content from container."""
        return await self.execute_command(f"cat {file_path}")

    async def write_file(self, file_path: str, content: str) -> Tuple[bool, str]:
        """Write content to file in container."""
        # Escape content for shell
        escaped_content = content.replace("'", "'\\''")
        output, exit_code = await self.execute_command(
            f"cat > {file_path} << 'EOFMARKER'\n{content}\nEOFMARKER"
        )
        if exit_code != 0:
            return False, f"Failed to write file: {output}"
        return True, "File written successfully"

    async def list_files(self, directory: str = ".") -> Tuple[str, int]:
        """List files in directory."""
        return await self.execute_command(f"find {directory} -type f -name '*.py' | head -100")

    def get_container_id(self) -> Optional[str]:
        """Get the container ID."""
        return self.container_id

    async def cleanup(self):
        """Clean up container and temporary files."""
        if self.container_id:
            try:
                # Stop and remove container
                await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "rm", "-f", self.container_id],
                    capture_output=True,
                    timeout=30,
                )
                logger.info(f"Container removed: {self.container_id[:12]}")
            except Exception as e:
                logger.warning(f"Failed to remove container: {e}")
            finally:
                self.container_id = None
        
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temp dir: {e}")
            finally:
                self._temp_dir = None
