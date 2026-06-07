# DeLM: Decentralized Language Models

DeLM replaces centralized orchestration with a shared, verified context and task queue, enabling parallel agents to asynchronously accumulate reusable progress for scalable, reliable, and cost-efficient test-time reasoning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/yuzhenmao/DeLM/blob/main/LICENSE) 
### [Project](https://yuzhenmao.github.io/DeLM/) | [Paper](https://arxiv.org/abs/2606.10662)

## What DeLM Does

For one SWE-bench task, DeLM runs `n_solvers` solver threads in parallel:

- Each thread owns one `PlannerAgent`, one or more delegated
  `SWEBenchImplementerAgent` runs, one Docker container, and one local memory.
- Each thread can write compact typed notes into `SharedLessons`.
- Peer threads read the shared notes before planning or implementing.
- The final `pass@N` score is the oracle best result among the `N` parallel
  solver threads. The `avg@1` score is the per-thread success rate and is the
  leaderboard-comparable metric.

The important design split is:

| Component | Scope | Purpose |
|---|---:|---|
| Local memory | One solver thread | Keeps that thread's own observations, actions, and results. It is compacted so long trajectories do not dominate the prompt. |
| Shared lessons | All solver threads for one task | Stores compact cross-thread facts, failed attempts, observations, claims, and patch summaries so peers can avoid duplicated work. |

## File Structure

```text
.
|-- bench_swebench.py
|   Main CLI entry point for SWE-bench runs.
|
|-- config/
|   |-- example/model_config.yaml
|   |   Template for private model/API configuration. Copy this to
|   |   config/model_config.yaml, which is gitignored.
|   `-- presets/swebench_gemini.yaml
|       Default Gemini-3-Flash SWE-bench preset.
|
|-- src/
|   |-- agents/
|   |   |-- planner_agent.py
|   |   `-- swebench_implementer_agent.py
|   |-- prompts/
|   |   |-- note_rules.py
|   |   |-- swebench_global.py
|   |   `-- swebench_local.py
|   |-- runners/
|   |   |-- swebench_orchestrator.py
|   |   |-- solver_thread.py
|   |   `-- implementer_runner.py
|   |-- results/
|   |   |-- csv_writer.py
|   |   `-- trajectory.py
|   |-- tools/
|   |   |-- delegate.py
|   |   |-- submit.py
|   |   `-- trace_formatter.py
|   |-- metrics/
|   |   `-- divergence.py
|   |       Cross-thread trajectory divergence metrics.
|   |-- common/
|   |   `-- utils.py
|   |       Shared utility helpers (JSON parsing, etc.).
|   |-- config.py
|   |-- memory_compactor.py
|   |-- shared_lessons.py
|   |-- verifier.py
|   |-- modes.py
|   `-- swebench_modes.py
|
|-- benchmark/
|   |-- bench_swebench.py
|   |-- benchmark.py
|   |-- common/
|   |   Shared runner/environment interfaces still used by the SWE-bench code.
|   |-- swebench/
|   |   SWE-bench data loading, Docker execution, ACI tools, and grading.
|   `-- long_source/
|       Long-source splitting, summarizing, unfolding, and verification.
|
|-- base/
|   |-- agent/
|   |   Shared base agent/action/memory abstractions.
|   `-- engine/
|       Async OpenAI-compatible LLM client, event logging, and cost tracking.
|
|-- workspace/
|   Runtime output directory. It is gitignored.
|
|-- website/
|   Project page assets (figures and demo video).
|
|-- requirements.txt
`-- .env.example
```

## Environment Setup

### 1. System Requirements

- Linux or another Docker-capable environment.
- Docker daemon running and accessible by the current user.
- Python 3.11+.
- Network access for Hugging Face dataset loading, SWE-bench Docker images, and
  LLM API calls.

### 2. Create an Environment

```bash
cd /home/harrymao/projects/DeLM

conda create -n delm python=3.11 -y
conda activate delm

pip install -r requirements.txt
```

If you prefer not to use conda, create any Python 3.11+ virtual environment and
install the same requirements.

### 3. Configure Model Access

The recommended route is a private YAML file:

```bash
cp config/example/model_config.yaml config/model_config.yaml
```

Then edit `config/model_config.yaml`:

```yaml
models:
  "google/gemini-3-flash-preview":
    api_type: "openai"
    base_url: "https://openrouter.ai/api/v1"
    api_key: "<your API key>"
```

`config/model_config.yaml` is gitignored and should never be committed.

You can also use environment variables. This is useful for CI or temporary
runs:

```bash
export AUTOENV_OPENAI_MODELS="google/gemini-3-flash-preview"
export AUTOENV_OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export AUTOENV_OPENAI_API_KEY="<your API key>"

# If routing through OpenRouter, this provider-specific key is also accepted.
export OPENROUTER_API_KEY="$AUTOENV_OPENAI_API_KEY"
```

Provider-specific environment variables can override YAML keys:

| Provider route | Environment variable |
|---|---|
| OpenRouter | `OPENROUTER_API_KEY` |
| Google Gemini direct API | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Generic OpenAI-compatible fallback | `AUTOENV_OPENAI_API_KEY` or `OPENAI_API_KEY` |

### 4. Check Docker

```bash
docker ps
```

If the command fails, fix Docker access before running the benchmark. DeLM uses
official SWE-bench Docker images and creates one container per solver thread.

## Quick Smoke Tests

### Single Task

```bash
python bench_swebench.py \
  --config config/presets/swebench_gemini.yaml \
  --tasks django__django-16333 \
  --max_concurrency 1
```

### Multiple Specific Tasks

```bash
python bench_swebench.py \
  --config config/presets/swebench_gemini.yaml \
  --tasks sympy__sympy-14976,django__django-11490,sphinx-doc__sphinx-7454,psf__requests-2931 \
  --max_concurrency 4
```

### First 20 Tasks From the Deterministic Test Subset

Create a local smoke config under the ignored `workspace/` directory:

```bash
mkdir -p workspace
cp config/presets/swebench_gemini.yaml workspace/smoke20_gemini.yaml
```

Edit `workspace/smoke20_gemini.yaml` and add or update:

```yaml
max_tasks: 20
result_folder: workspace/logs/smoke20_gemini
```

Then run:

```bash
python bench_swebench.py \
  --config workspace/smoke20_gemini.yaml \
  --max_concurrency 4
```


## Running a Benchmark

The shipped preset loads the full SWE-bench Verified test split:

```yaml
dataset_name: princeton-nlp/SWE-bench_Verified
split: test
```

With no `max_tasks` and no `--tasks` filter, this runs the full split. At the
time of writing, SWE-bench Verified contains 500 test tasks.

```bash
python bench_swebench.py \
  --config config/presets/swebench_gemini.yaml \
  --max_concurrency 4
```

For pass@4-style runs, copy the preset and set:

```yaml
n_solvers: 4
result_folder: workspace/logs/swebench_gemini_n4
```

Then run the copied config. Keep each run's `result_folder` unique so
`results.csv`, trajectories, and events are not mixed across experiments.

To run a fixed task list, use `--tasks`. This ignores `max_tasks` truncation:

```bash
python bench_swebench.py \
  --config config/presets/swebench_gemini.yaml \
  --tasks django__django-11490,scikit-learn__scikit-learn-25931 \
  --max_concurrency 2
```

To resume a run after interruption:

```bash
python bench_swebench.py \
  --config workspace/smoke20_gemini.yaml \
  --skip-completed \
  --max_concurrency 4
```

`--skip-completed` skips rows already marked `success=True` in the target
`results.csv`.

To summarize one or more result folders and separate infrastructure zero-attempt
rows from wall-clock-capped partial runs:

```bash
python benchmark/swebench/summarize_results.py workspace/logs
```

## Outputs

Each run writes to `result_folder`:

| Path | Meaning |
|---|---|
| `results.csv` | One row per task with success, avg@1 thread count, pass@N oracle success, cost, tokens, wall-clock time, forced-cap flags, and winning thread. |
| `events.jsonl` | One JSON object per completed LLM call, including model, role, token usage, cost, latency, and task/thread id. |
| `trajectories/` | Per-task trajectory files for debugging agent behavior. |
| `lessons/` | Per-task shared lesson JSONL files. |
| task-level Docker logs | SWE-bench execution and grading logs. |

Important metrics:

| Metric | Interpretation |
|---|---|
| `avg@1` | Per-thread success rate. This is the single-attempt comparable metric. |
| `pass@N` | Oracle best-of-N success. This measures test-time scaling but assumes the evaluator can pick the best parallel thread by reward. |
| `cost` | Sum of recorded LLM-call costs for the task. |
| `forced_cap` | The task hit a wall-clock or token cap. |
| `forced_submit` | A thread was forced to submit because a budget/cap was reached. |

## Reported SWE-bench Gemini Results

We report the following SWE-bench Verified results with Gemini-3-Flash as the base model:

| Method | Avg.@1 | Pass@2 | Pass@4 | Cost/Task |
|---|---:|---:|---:|---:|
| mini-SWE-agent | 54.7% | 65.6% | 75.1% | $0.26 |
| Claude Code | 49.3% | 57.1% | 66.3% | -- |
| AOrchestra | 55.2% | 64.5% | 73.2% | $0.24 |
| AOrchestra-Parallel | 56.4% | 63.2% | 71.8% | $0.25 |
| **DeLM** | **65.7%** | **72.9%** | **77.4%** | **$0.12** |

The same draft reports DeLM cost at about `$0.12` per SWE-bench task, roughly
half the baseline cost in that comparison.


## Citation

```bibtex
@misc{mao2026delm,
      title={Decentralized Multi-Agent Systems with Shared Context}, 
      author={Yuzhen Mao and Azalia Mirhoseini},
      year={2026},
      eprint={2606.10662},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2606.10662}, 
}
```