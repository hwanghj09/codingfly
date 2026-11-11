"""Capture the unchanged legacy test/benchmark baseline with provenance."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

from evaluation.report import provenance, write_report


ROOT = Path(__file__).resolve().parents[1]


def run(output):
    records = []
    for arguments in (("-m", "unittest", "-v", "test_fly", "test_goal", "test_pseudocode"), ("benchmark.py",)):
        started = time.perf_counter()
        result = subprocess.run([sys.executable, *arguments], cwd=ROOT, capture_output=True, text=True, timeout=120)
        records.append({"command": ["python", *arguments], "exit_code": result.returncode,
                        "seconds": time.perf_counter() - started, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError("Legacy baseline failed: " + result.stderr)
    report = {"version": 1, "mode": "legacy", "provenance": provenance(ROOT), "runs": records}
    write_report(Path(output) / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/baseline")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
