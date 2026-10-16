# External tasks through the shared DeLM runner

`delm.py --task-config /path/to/task.json` runs externally supplied tasks through
`runtime/runner.py`, the same provider workers, board, collection and reporting
as Terminal-Bench. No math-specific execution engine or scheduler is needed.
Existing positional Terminal-Bench commands and official grading remain unchanged.

## Configuration

Keep experiment inputs outside version control (for example in `.local/`). A
configuration describes the task, not the model or number of agents:

```json
{
  "name": "odd-sum",
  "image": "sha256:<64 hexadecimal digits from docker image inspect>",
  "instruction": "problem.txt",
  "files": {"PROBLEM.txt": "problem.txt"},
  "prompt": "collaboration.txt",
  "rubric": "rubric.txt",
  "cpus": 2,
  "memory_mb": 4096,
  "storage_mb": 1024,
  "timeout_seconds": 180,
  "network": "model-only",
  "timing": "synchronized",
  "evaluator": "review"
}
```

The image must already exist and contain the pinned provider CLI and Humanize
runtime expected by the existing Dockerfiles. Its working directory must be a
non-root task workspace. Runtime preflight checks CLI/driver/authentication before
any model calls. No image build or download happens implicitly.

Paths are relative to the configuration directory. Only explicitly named input
files are read; traversal and input symlinks are rejected. `files` maps relative
workspace destinations to input files. Configuration, instruction, prompt, rubric
and initial files are snapshotted and hashed under the run's `task-source/` folder.
Only `files` entries are copied into worker workspaces; the rubric remains on the
controller. The image and inputs must not contain reference solutions or secrets.
Changing source, inputs or settings requires a new series; existing repeats are
never overwritten or automatically retried.

`prompt` and `rubric` are optional. Without a prompt override, the existing n=2/n=4
DeLM collaboration prompts are used unchanged. An override is the complete worker
prompt, with `{instruction}`, `{worker}`, `{roster}`, `{count}`, and `{workspace}`
placeholders. It should specify the final deliverable, use of `/tools/context.sh`,
and prohibit extra agents, credential/record access, and external solution lookup.
Use one template across agent counts when testing count-only scaling. Existing
built-in n=2/n=4 prompts differ in their initial role instructions.

## Launch and inspect

```sh
python3.13 delm.py --task-config /path/to/task.json --n 2 --harness codex \
  --auth-mode coding-plan --auth-file ~/.codex/auth.json \
  --model gpt-6-astra --series math-smoke --repeat 1 --dry-run
# Remove --dry-run to execute on an idle Linux Docker host.
python3.13 inspect_run.py .local/runs/math-smoke/odd-sum/repeat-1
python3.13 -m reporting.report
```

Model, authentication and worker count use the existing launcher options. External
tasks support the same n=2/n=4 DeLM peer mode; single-agent/native-subagent baselines,
eight-agent execution and multi-bundle scheduling are outside this change. External
tasks currently do not configure Compose sidecar services.

## Policies and evaluation

- `network`: `model-only` (default) uses an internal Docker network and an HTTPS
  proxy restricted to provider/authentication hosts; preflight checks that direct
  internet access is blocked. `unrestricted` uses the existing task networking.
- `timing`: `synchronized` (default) waits for prepared sessions and records a common
  release/deadline. Reported execution seconds start at controller release. With
  `per-worker`, each session gets its own time budget and reports use the earliest
  model-turn start, as existing Terminal-Bench runs do. Setup, collection and grading
  are excluded from these execution measurements. Every new run records its policy;
  historical records are neither rewritten nor treated as having synchronized starts.
- `evaluator`: `review` captures each workspace archive, final answer and a review
  packet. `finished` means execution and collection succeeded, **not** that the proof
  is correct. Correctness and first-success timing remain unknown until reviewed.
  Review JSON is an export for an external reviewer; editing it does not automatically
  change run records or report grades. Identity mappings stay outside the packet;
  verbatim answers can still reveal collaboration.

Native trajectories, usage and failure records use the shared runtime. Incomplete
runs retain observations and do not get complete aggregate summaries. Token records
are not dollar invoices. Lean environments may be supplied as task images, but the
separate local Erdős replay/semantic/axiom verification pipeline is not included;
compilation alone must not be presented as full proof verification.

Old pilot bundles require their archived code. This interface does not relabel or
re-run historical trials. All historical configurations and results remain outside
this branch.

## Offline checks

```sh
python3.13 -B -m unittest discover -s tests -t . -v
```
