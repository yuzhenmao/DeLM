# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

"""Result reporter for SWE-bench benchmark."""
from pathlib import Path
from typing import Dict

from benchmark.common.runner import LevelResult


def print_results(
    results: Dict[str, LevelResult],
    csv_path: Path,
    trajectory_folder: Path,
):
    """Calculate and display SWE-bench benchmark results."""
    total_reward = sum(r.total_reward for r in results.values())
    num_tasks = len(results)
    avg_reward = total_reward / num_tasks if num_tasks > 0 else 0.0
    cost = sum(r.cost for r in results.values())

    oracle_resolved = sum(1 for r in results.values() if r.total_reward > 0)
    oracle_rate = oracle_resolved / num_tasks if num_tasks > 0 else 0.0

    threads_passed = sum((getattr(r, "n_solvers_passed", 0) or 0) for r in results.values())
    threads_total = sum((getattr(r, "n_solvers_total", 0) or 0) for r in results.values())
    avg1_rate = threads_passed / threads_total if threads_total > 0 else 0.0

    # Calculate token usage
    total_input_tokens = sum(r.input_tokens for r in results.values())
    total_output_tokens = sum(r.output_tokens for r in results.values())
    avg_steps = sum(r.steps for r in results.values()) / num_tasks if num_tasks > 0 else 0.0

    print("\n" + "=" * 70)
    print("SWE-bench Verified Results")
    print("=" * 70)
    print(f"Total Instances: {num_tasks}")
    print(f"avg@1 (per-thread, leaderboard-comparable): "
          f"{threads_passed}/{threads_total} = {avg1_rate:.2%}")
    print(f"pass@N (ORACLE best-of-{max((getattr(r, 'n_solvers_total', 1) or 1) for r in results.values()) if results else 1}, "
          f"upper bound only): {oracle_resolved}/{num_tasks} = {oracle_rate:.2%}")
    print("-" * 70)
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Total Input Tokens: {total_input_tokens:,}")
    print(f"Total Output Tokens: {total_output_tokens:,}")
    print(f"Total Cost: ${cost:.4f}")
    print("-" * 70)
    print(f"Results saved to: {csv_path}")
    print(f"Trajectories saved to: {trajectory_folder}")
    print("=" * 70)


def generate_summary_report(
    results: Dict[str, LevelResult],
    output_path: Path,
):
    """Generate a detailed summary report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# SWE-bench Verified Summary Report\n\n")
        
        # Overall statistics
        num_tasks = len(results)
        oracle_resolved = sum(1 for r in results.values() if r.total_reward > 0)
        threads_passed = sum((getattr(r, "n_solvers_passed", 0) or 0) for r in results.values())
        threads_total = sum((getattr(r, "n_solvers_total", 0) or 0) for r in results.values())
        oracle_rate = oracle_resolved / num_tasks * 100 if num_tasks > 0 else 0.0
        avg1_rate = threads_passed / threads_total * 100 if threads_total > 0 else 0.0

        f.write("## Overall Statistics\n\n")
        f.write(f"- Total Instances: {num_tasks}\n")
        f.write(f"- **avg@1 (per-thread, leaderboard-comparable):** "
                f"{threads_passed}/{threads_total} = {avg1_rate:.1f}%\n")
        f.write(f"- **pass@N (ORACLE best-of-N, upper bound only):** "
                f"{oracle_resolved}/{num_tasks} = {oracle_rate:.1f}%\n\n")

        # Per-instance results
        f.write("## Per-Instance Results\n\n")
        f.write("| Instance ID | Oracle Status | Threads Passed | Steps | Cost |\n")
        f.write("|-------------|---------------|----------------|-------|------|\n")

        for instance_id, result in sorted(results.items()):
            status = "✓ Resolved" if result.total_reward > 0 else "✗ Failed"
            tp = f"{getattr(result, 'n_solvers_passed', 0)}/{getattr(result, 'n_solvers_total', 0)}"
            f.write(f"| {instance_id} | {status} | {tp} | {result.steps} | ${result.cost:.4f} |\n")
        
        f.write("\n")
    
    return output_path

