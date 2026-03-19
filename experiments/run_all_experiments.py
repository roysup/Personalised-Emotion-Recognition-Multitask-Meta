"""
Run All Experiments — Sequential Execution
Runs each experiment script one after the other via subprocess.
Logs stdout/stderr per experiment and
prints a summary with pass/fail and elapsed time at the end.
"""
import os
import sys
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ordered list of experiment scripts to run
EXPERIMENTS = [
    # MTL Baselines
    ('MTL_baselines', 'pstl.py'),
    ('MTL_baselines', 'stl.py'),
    ('MTL_baselines', 'mtl_hps.py'),
    ('MTL_baselines', 'mtl_pcgrad.py'),
    ('MTL_baselines', 'mtl_uw.py'),
    # MTML Baselines
    ('MTML_baselines', 'si.py'),
    ('MTML_baselines', 'tlft.py'),
    ('MTML_baselines', 'transfer_mtl.py'),
    ('MTML_baselines', 'mtl_retrain.py'),
    ('MTML_baselines', 'pure_meta.py'),
    ('MTML_baselines', 'reptile_st.py'),
    ('MTML_baselines', 'reptile_mt.py'),
    ('MTML_baselines', 'reptile_mi.py'),
]


def run_experiment(folder, script):
    """Run a single experiment script and return (success, elapsed_seconds)."""
    script_path = os.path.join(REPO_ROOT, 'experiments', folder, script)
    label = f"{folder}/{script}"
    print(f"\n{'='*70}")
    print(f"  STARTING: {label}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=REPO_ROOT,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  COMPLETED: {label}  ({elapsed:.1f}s)")
    else:
        print(f"\n  FAILED: {label}  (exit code {result.returncode}, {elapsed:.1f}s)")

    return result.returncode == 0, elapsed


if __name__ == '__main__':
    total_t0 = time.time()
    summary = []

    for folder, script in EXPERIMENTS:
        success, elapsed = run_experiment(folder, script)
        summary.append((f"{folder}/{script}", success, elapsed))

    total_elapsed = time.time() - total_t0

    # Print summary table
    print(f"\n\n{'='*70}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<45} {'Status':<10} {'Time':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10}")
    for label, success, elapsed in summary:
        status = "PASS" if success else "FAIL"
        print(f"  {label:<45} {status:<10} {elapsed:>9.1f}s")
    print(f"  {'-'*45} {'-'*10} {'-'*10}")
    passed = sum(1 for _, s, _ in summary if s)
    print(f"  {'TOTAL':<45} {passed}/{len(summary):<10} {total_elapsed:>9.1f}s")
    print(f"{'='*70}")
