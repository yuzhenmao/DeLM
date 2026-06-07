# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

"""SWE-bench Verified benchmark implementation with ACI tools."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from base.engine.logs import logger
from base.agent.base_agent import BaseAgent
from benchmark.benchmark import Benchmark, LevelSpec
from benchmark.common.env import Action, BasicInfo, Environment, Observation
from benchmark.common.incremental_runner import IncrementalRunner
from benchmark.common.runner import LevelResult, StepRecord
from benchmark.swebench.data_loader import SWEBenchDataLoader, SWEBenchInstance
from benchmark.swebench.swebench_executor import SWEBenchExecutor
from benchmark.swebench.utils import resolve_path
from benchmark.swebench.aci_tools import ACIToolManager, format_command_output

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/example/benchmarks/swebench.yaml"

INCLUDE_HINTS = False


@dataclass
class SWEBenchConfig:
    """Configuration for SWE-bench Verified benchmark."""

    dataset_name: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"
    subset_seed: Optional[int] = None
    subset_sizes: Optional[Dict[str, int]] = None
    subset_role: Optional[str] = None
    max_steps: int = 50
    max_tasks: Optional[int] = None
    docker_timeout: int = 1800
    grading_timeout: int = 600
    model: Optional[str] = None
    result_folder: Path = PROJECT_ROOT / "workspace/logs"
    trajectory_dir: Optional[Path] = None
    csv_summary_path: Optional[Path] = None
    timestamp: Optional[str] = None
    env_init: Optional[dict[str, str]] = None
    cache_dir: Optional[str] = None  # HuggingFace cache directory
    window_size: int = 100  # ACI window size for file viewing

    @classmethod
    def load(cls, config_path: Path | str = DEFAULT_CONFIG_PATH) -> "SWEBenchConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        dataset_name = raw.get("dataset_name", "princeton-nlp/SWE-bench_Verified")
        split = raw.get("split", "test")
        subset_seed = raw.get("subset_seed")
        if subset_seed is not None:
            subset_seed = int(subset_seed)

        subset_sizes = raw.get("subset_sizes")
        if subset_sizes:
            subset_sizes = {
                str(k): int(v)
                for k, v in subset_sizes.items()
                if v is not None
            }
        else:
            subset_sizes = None
        subset_role = raw.get("subset_role")
        max_steps = int(raw.get("max_steps", 50))
        max_tasks = raw.get("max_tasks")
        if max_tasks is not None:
            max_tasks = int(max_tasks)

        docker_timeout = int(raw.get("docker_timeout", 1800))
        grading_timeout = int(raw.get("grading_timeout", 600))
        model = raw.get("model")
        env_init = raw.get("env_init")
        cache_dir = raw.get("cache_dir")
        window_size = int(raw.get("window_size", 100))

        result_folder = resolve_path(
            raw.get("result_folder", "workspace/logs"), config_path, PROJECT_ROOT
        )

        trajectory_dir = raw.get("trajectory_dir")
        if trajectory_dir:
            trajectory_dir = resolve_path(trajectory_dir, config_path, PROJECT_ROOT)

        csv_summary_path = raw.get("csv_summary_path")
        if csv_summary_path:
            csv_summary_path = resolve_path(csv_summary_path, config_path, PROJECT_ROOT)

        timestamp = raw.get("timestamp")

        return cls(
            dataset_name=dataset_name,
            split=split,
            subset_seed=subset_seed,
            subset_sizes=subset_sizes,
            subset_role=subset_role,
            max_steps=max_steps,
            max_tasks=max_tasks,
            docker_timeout=docker_timeout,
            grading_timeout=grading_timeout,
            model=str(model) if model is not None else None,
            result_folder=result_folder or PROJECT_ROOT / "workspace/logs",
            trajectory_dir=trajectory_dir,
            csv_summary_path=csv_summary_path,
            timestamp=timestamp,
            env_init=env_init,
            cache_dir=cache_dir,
            window_size=window_size,
        )


class SWEBenchEnvironment(Environment):
    """Environment for a single SWE-bench instance with ACI tools."""

    def __init__(
        self,
        level: LevelSpec,
        config: SWEBenchConfig,
        instance: SWEBenchInstance,
        name_suffix: str = "",
        auto_grade_on_max_steps: bool = True,
    ):
        self.instance_id: str = level["id"]
        self.instance = instance
        self.config = config
        self.name_suffix = str(name_suffix or "")
        self._auto_grade_on_max_steps = bool(auto_grade_on_max_steps)

        # State
        self._steps = 0
        self._done = False
        self._command_history: List[str] = []
        self._patch_submitted: bool = False

        if config.timestamp:
            folder_name = f"{self.instance_id}{self.name_suffix}_{config.timestamp}"
        else:
            folder_name = f"{self.instance_id}{self.name_suffix}"
        self._logs_dir = config.result_folder / folder_name / "logs"
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        # Create executor
        self._executor = SWEBenchExecutor(
            instance=self.instance,
            logs_dir=self._logs_dir,
            timeout=config.docker_timeout,
            env_init=config.env_init,
            name_suffix=self.name_suffix,
            grading_timeout=getattr(config, "grading_timeout", 600),
        )
        
        # ACI Tool Manager (initialized after container starts)
        self._aci_manager: Optional[ACIToolManager] = None

    def get_basic_info(self) -> BasicInfo:
        """Get basic information about the task."""
        # Get command docs from ACI manager if available
        command_docs = ""
        if self._aci_manager:
            command_docs = self._aci_manager.get_command_docs()
        else:
            # Fallback before container starts
            command_docs = self._get_default_command_docs()
        
        return BasicInfo(
            env_id=self.instance_id,
            instruction=self._build_instruction(),
            action_space=self._get_action_space(),
            max_steps=self.config.max_steps,
            meta_data={
                "repo": self.instance.repo,
                "base_commit": self.instance.base_commit,
                "version": self.instance.version,
                "instance_id": self.instance_id,
                "command_docs": command_docs,
            },
        )

    def _get_default_command_docs(self) -> str:
        """Get default command documentation before ACI manager is initialized."""
        return """Available commands:

=== FILE VIEWING ===
open <file> [<line_number>]
    Opens the file at the given path. If line_number is provided, the window will start at that line.
    
scroll_down [<lines>]
    Moves the window down by the specified number of lines (default: window_size).
    
scroll_up [<lines>]
    Moves the window up by the specified number of lines (default: window_size).
    
goto <line_number>
    Moves the window to show the specified line number at the top.

=== FILE EDITING (use str_replace for best results) ===
str_replace <file_path>
exact text to find (must be unique in file)
    **PREFERRED METHOD** - Replaces exact text match. The search text must be unique.
    Use this for most edits - it's more reliable than line-based edit.
    
edit <start_line>:<end_line>
<replacement_text>
end_of_edit
    Replaces lines from start_line to end_line (inclusive) with the given text.
    For Python files, syntax checking is performed automatically.
    NOTE: For large edits, prefer str_replace to avoid command length limits.

delete_lines <file_path> <start_line> <end_line>
    Deletes lines from start_line to end_line (inclusive) in a single operation.
    **USE THIS** instead of running sed multiple times to delete a block of code.
    Example: delete_lines myfile.py 100 120  (deletes lines 100-120)

insert <file_path> <line_number>
<content to insert>
end_of_insert
    Inserts content after the specified line number.

create <file_path>
<file content>
end_of_create
    Creates or overwrites a file with the given content.
    
=== SEARCHING ===
search_file <pattern> [<file>]
    Searches for CONTENT pattern in the current open file (or specified file).
    Shows matching lines with line numbers. Use this to find function definitions.
    
search_dir <pattern> [<directory>]
    Searches for CONTENT pattern in all files within the directory.
    Returns file names containing the pattern. Use this to find which file contains a function.
    
find_file <name_pattern> [<directory>]
    Finds files by FILE NAME pattern (not content). 
    Example: find_file test to find files with "test" in their name.

=== BASH COMMANDS ===
You can also run any bash command directly (e.g., ls, cat, grep, python, pytest, etc.)

TIP: For deleting code blocks, use "delete_lines <file> <start> <end>" instead of multiple sed commands.
"""

    def _build_instruction(self) -> str:
        """Build instruction from problem statement."""
        instruction = f"""You are a software engineer working on fixing a GitHub issue.

## Repository
{self.instance.repo}

## Issue Description
{self.instance.problem_statement}

## Your Task
1. Understand the issue described above
2. Explore the codebase to locate the relevant files
3. Make the necessary code changes to fix the issue
4. Ensure your fix doesn't break existing functionality

## Your Working Environment
- You are ALREADY running INSIDE a Docker container - do NOT try to run docker commands
- The repository is already cloned at `/testbed` and checked out to the correct commit
- Use ACI commands (open, find_file, search_dir) for file navigation
- Use the edit command for making changes (includes automatic syntax checking)
- When you're done, submit your fix using the 'submit' command

## Environment Setup
- To run Python commands (e.g., pytest, python scripts), you MUST first activate the conda environment:
  `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed`
- Basic shell commands (find, grep, cat, sed) do NOT require conda activation

## Running Tests
IMPORTANT: Check how this project runs tests BEFORE using pytest directly!
- Many projects have custom test runners (e.g., `runtests.py`, `manage.py test`)
- For Django projects: use `python tests/runtests.py <module>` instead of pytest
- For other projects: look for `runtests.py`, `setup.py test`, or check `README.md`/`CONTRIBUTING.md`
- Example for Django: `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && cd /testbed && python tests/runtests.py admin_checks`
"""
        # Off by default — hints_text is non-standard and leaks fix guidance.
        if INCLUDE_HINTS and self.instance.hints_text:
            instruction += f"\n## Hints\n{self.instance.hints_text}\n"

        return instruction

    def _get_action_space(self) -> str:
        """Get action space description."""
        return '''Use ACI commands or bash commands to explore, edit, and test code.

Primary ACI commands:
- open <file> [line]     : Open file with line numbers
- scroll_up/scroll_down  : Navigate within open file
- goto <line>            : Jump to specific line
- edit <start>:<end>     : Edit line range (with syntax check)
- search_file <pattern>  : Search in current file
- search_dir <pattern>   : Search in directory (returns file names only)
- find_file <name>       : Find files by name

You can also run any bash command directly.

Submit your fix when done:
- submit

Response format:
DISCUSSION
<your reasoning>
COMMAND
<single command>
'''

    async def reset(self, seed: int | None = None) -> Observation:
        """Reset environment by starting Docker container."""
        self._done = False
        self._steps = 0
        self._command_history = []
        self._patch_submitted = False
        self._container_started = False

        # Start container
        await self._executor.start_container()
        self._container_started = True  # Container is now running
        
        # Initialize ACI manager
        self._aci_manager = ACIToolManager(
            executor=self._executor,
            window_size=self.config.window_size
        )
        
        # Get initial state
        state_info = self._aci_manager.state.to_status_line()

        return {
            "message": "Environment ready. Repository cloned and checked out to base commit.",
            "output": f"Environment initialized.\n{state_info}\n\nStart by exploring the repository with 'find_file' or 'ls /testbed'.",
            "state_info": state_info,
            "current_step": 0,
            "max_steps": self.config.max_steps,
            "repo": self.instance.repo,
            "base_commit": self.instance.base_commit,
        }

    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute action and return observation, reward, done, info."""
        if self._done:
            raise RuntimeError("Environment already finished. Call reset() first.")

        self._steps += 1
        action_type = action.get("action", "")
        params = action.get("params", {})

        # Handle ACI commands (new format from SWEAgent)
        if action_type == "aci_command":
            command = params.get("command", "")
            if not command:
                return (
                    {"error": "No command provided", "state_info": self._get_state_info()},
                    0.0,
                    False,
                    {"error": "No command provided"},
                )

            self._command_history.append(command)
            
            # Execute via ACI manager
            output, exit_code = await self._aci_manager.execute(command)
            state_info = self._aci_manager.state.to_status_line()

            # Save command log
            self._log_command(command, output, exit_code)

            observation = {
                "command": command,
                "output": output,
                "exit_code": exit_code,
                "state_info": state_info,
                "current_step": self._steps,
                "max_steps": self.config.max_steps,
            }

            if self._steps >= self.config.max_steps:
                if self._auto_grade_on_max_steps:
                    self._done = True
                    reward, test_results = await self._executor.run_tests()
                    observation["message"] = "Max steps reached, running tests"
                    return observation, reward, self._done, {"max_steps_reached": True, "test_results": test_results}
                observation["message"] = "Max steps reached (implementer handoff; grading deferred to submit)"
                return observation, 0.0, False, {"max_steps_reached": True, "implementer_handoff": True}

            return observation, 0.0, self._done, {"command_executed": command}

        # Handle legacy execute action (backward compatibility)
        elif action_type == "execute":
            command = params.get("command", "")
            if not command:
                return (
                    {"error": "No command provided", "state_info": self._get_state_info()},
                    0.0,
                    False,
                    {"error": "No command provided"},
                )

            self._command_history.append(command)
            
            # Use ACI manager if available, otherwise direct execution
            if self._aci_manager:
                output, exit_code = await self._aci_manager.execute(command)
                state_info = self._aci_manager.state.to_status_line()
            else:
                output, exit_code = await self._executor.execute_command(command)
                output = format_command_output(output, exit_code)
                state_info = "(Open file: n/a) (Current directory: /testbed)"

            # Save command log
            self._log_command(command, output, exit_code)

            observation = {
                "command": command,
                "output": output,
                "exit_code": exit_code,
                "state_info": state_info,
                "current_step": self._steps,
                "max_steps": self.config.max_steps,
            }

            if self._steps >= self.config.max_steps:
                if self._auto_grade_on_max_steps:
                    self._done = True
                    reward, test_results = await self._executor.run_tests()
                    observation["message"] = "Max steps reached, running tests"
                    return observation, reward, self._done, {"max_steps_reached": True, "test_results": test_results}
                observation["message"] = "Max steps reached (implementer handoff; grading deferred to submit)"
                return observation, 0.0, False, {"max_steps_reached": True, "implementer_handoff": True}

            return observation, 0.0, self._done, {"command_executed": command}

        elif action_type == "view_file":
            file_path = params.get("file_path", "")
            start_line = params.get("start_line", 1)
            
            # Use ACI open command
            if self._aci_manager:
                output, exit_code = await self._aci_manager.cmd_open(f"{file_path} {start_line}")
                state_info = self._aci_manager.state.to_status_line()
            else:
                command = f"sed -n '{start_line},{start_line + 99}p' {file_path}"
                output, exit_code = await self._executor.execute_command(command)
                state_info = f"(Open file: {file_path}) (Current directory: /testbed)"
            
            observation = {
                "file_path": file_path,
                "content": output,
                "output": output,
                "exit_code": exit_code,
                "state_info": state_info,
                "current_step": self._steps,
                "max_steps": self.config.max_steps,
            }
            
            return observation, 0.0, False, {"file_viewed": file_path}

        elif action_type == "edit_file":
            file_path = params.get("file_path", "")
            old_content = params.get("old_content", "")
            new_content = params.get("new_content", "")
            
            # Read current file
            current_content, _ = await self._executor.get_file_content(file_path)
            
            # Replace content
            if old_content in current_content:
                updated_content = current_content.replace(old_content, new_content, 1)
                success, msg = await self._executor.write_file(file_path, updated_content)
                
                # Syntax check for Python files
                lint_msg = ""
                if file_path.endswith('.py'):
                    lint_output, lint_code = await self._executor.execute_command(
                        f"python -m py_compile {file_path} 2>&1"
                    )
                    if lint_code != 0:
                        lint_msg = f"\n[Syntax Error]\n{lint_output}"
                    else:
                        lint_msg = "\n[Syntax check passed]"
                
                observation = {
                    "file_path": file_path,
                    "success": success,
                    "message": msg + lint_msg,
                    "output": msg + lint_msg,
                    "state_info": self._get_state_info(),
                    "current_step": self._steps,
                    "max_steps": self.config.max_steps,
                }
            else:
                observation = {
                    "file_path": file_path,
                    "success": False,
                    "message": "Old content not found in file",
                    "output": "Error: Old content not found in file",
                    "state_info": self._get_state_info(),
                    "current_step": self._steps,
                    "max_steps": self.config.max_steps,
                }
            
            return observation, 0.0, False, {"file_edited": file_path}

        elif action_type == "submit":
            reward, test_results = await self._executor.run_tests()
            self._done = True
            self._patch_submitted = True

            observation = {
                "message": "Fix submitted.",
                "output": "Fix submitted for evaluation.",
                "state_info": self._get_state_info(),
                "current_step": self._steps,
            }

            return observation, reward, True, {"submitted": True, "test_results": test_results}

        elif action_type == "error":
            # Handle parsing errors
            error_msg = params.get("message", "Unknown error")
            return (
                {
                    "error": error_msg,
                    "output": f"Error: {error_msg}. Please use the correct format:\nDISCUSSION\n<reasoning>\nCOMMAND\n<command>",
                    "state_info": self._get_state_info(),
                    "current_step": self._steps,
                    "max_steps": self.config.max_steps,
                },
                0.0,
                False,
                {"error": error_msg},
            )

        # === ImplementerAgent finish action ===
        elif action_type == "finish":
            # ImplementerAgent finish — report completion without running tests
            status = params.get("status", "done")
            message = params.get("message", "")
            completed = params.get("completed", [])
            issues = params.get("issues", [])

            return (
                {
                    "message": f"ImplementerAgent finished with status={status}",
                    "output": f"Finished: {message}\nCompleted: {completed}\nIssues: {issues}",
                    "status": status,
                    "completed": completed,
                    "issues": issues,
                    "state_info": self._get_state_info(),
                    "current_step": self._steps,
                },
                0.0,
                True,  # done=True for finish action
                {"finished": True, "finish_result": {"status": status, "message": message, "completed": completed, "issues": issues}},
            )

        else:
            return (
                {
                    "error": f"Unknown action type: {action_type}",
                    "output": f"Error: Unknown action type '{action_type}'",
                    "state_info": self._get_state_info(),
                },
                0.0,
                False,
                {"error": f"Unknown action type: {action_type}"},
            )

    def _get_state_info(self) -> str:
        """Get current state information."""
        if self._aci_manager:
            return self._aci_manager.state.to_status_line()
        return "(Open file: n/a) (Current directory: /testbed)"

    def _log_command(self, command: str, output: str, exit_code: int) -> None:
        """Log command execution to file."""
        command_log = self._logs_dir / "commands.log"
        with command_log.open("a", encoding="utf-8") as f:
            f.write(f"[Step {self._steps}] {command}\n")
            f.write(f"Exit Code: {exit_code}\n")
            f.write(f"Output:\n{output}\n")
            f.write("-" * 80 + "\n")

    async def close(self):
        """Close environment and cleanup resources."""
        await self._executor.cleanup()


class SWEBenchRunner(IncrementalRunner):
    """Runner for SWE-bench: incremental save + container cleanup."""

    async def run(self, agent: BaseAgent, env: Environment) -> LevelResult:
        result = None
        try:
            result = await super().run(agent, env)
            return result
        except Exception as e:
            logger.error(f"SWE-bench task failed: {type(e).__name__}: {e}")
            info = env.get_basic_info()
            result = LevelResult(
                model=getattr(getattr(agent.llm, "config", None), "model", "unknown"),
                total_reward=0.0,
                steps=0,
                done=False,
                trace=[StepRecord(
                    observation={"error": str(e)},
                    action={"error": "task_failed"},
                    reward=0.0,
                    raw_response="",
                    done=False,
                    info={"error": str(e), "error_type": type(e).__name__},
                )],
                cost=0.0,
            )
            if self.csv_summary_path:
                self._append_csv_row(info.env_id, result)
            return result
        finally:
            # Cleanup containers
            if hasattr(env, 'close'):
                try:
                    await env.close()
                except Exception as cleanup_error:
                    logger.error(f"Container cleanup failed: {cleanup_error}")


class SWEBenchBenchmark(Benchmark):
    """SWE-bench Verified benchmark."""

    def __init__(self, config: SWEBenchConfig):
        self.config = config
        self._data_loader = SWEBenchDataLoader(
            dataset_name=config.dataset_name,
            split=config.split,
            cache_dir=config.cache_dir,
            subset_seed=config.subset_seed,
            subset_sizes=config.subset_sizes,
            subset_role=config.subset_role,
        )
        self._instances: Dict[str, SWEBenchInstance] = {}
        
        # Setup runner
        base_dir = config.result_folder.parent if config.result_folder.name == "results" else config.result_folder
        trajectory_dir = config.trajectory_dir or (base_dir / "trajectories")
        csv_summary_path = config.csv_summary_path or (base_dir / "results.csv")
        self._runner = SWEBenchRunner(
            trajectory_dir=trajectory_dir,
            csv_summary_path=csv_summary_path
        )

    def list_levels(self) -> List[LevelSpec]:
        """List all available instances."""
        instances = self._data_loader.load_instances()
        
        levels = []
        for inst in instances:
            self._instances[inst.instance_id] = inst
            levels.append({
                "id": inst.instance_id,
                "_instance": inst,
            })
            
            if self.config.max_tasks and len(levels) >= self.config.max_tasks:
                break

        logger.info(f"Loaded {len(levels)} SWE-bench instances")
        return levels

    def make_env(self, level: LevelSpec) -> Environment:
        """Create environment for a specific instance."""
        instance = level.get("_instance") or self._instances.get(level["id"])
        if not instance:
            raise ValueError(f"Instance not found: {level['id']}")
        return SWEBenchEnvironment(level, self.config, instance)

    async def run(self, agent_cls, agent_kwargs=None, runner=None, **kwargs):
        """Run benchmark with SWEBenchRunner as default."""
        runner = runner or self._runner
        return await super().run(agent_cls, agent_kwargs, runner, **kwargs)


def load_benchmark(config_path: Path | str = DEFAULT_CONFIG_PATH) -> SWEBenchBenchmark:
    """Load SWE-bench benchmark from config file."""
    cfg = SWEBenchConfig.load(config_path)
    return SWEBenchBenchmark(cfg)
