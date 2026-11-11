"""Save compact, reviewable research evidence; leave weights/full logs in artifacts/."""

import argparse
import json
from pathlib import Path

from evaluation.report import provenance, write_report
from evaluation.sandbox import isolation_status


ROOT = Path(__file__).resolve().parents[1]


def snapshot(run_dir, output, *, test_log=None):
    run_dir, output = Path(run_dir), Path(output)
    baseline = json.loads((run_dir / "baseline/report.json").read_text())
    toy = json.loads((run_dir / "toy/report.json").read_text())
    code = json.loads((run_dir / "code/report.json").read_text())
    for index, run in enumerate(toy["runs"]):
        run.pop("history")
        if index:
            run.pop("example_trace")
    write_report(output / "baseline.json", baseline)
    write_report(output / "toy-p2.json", toy)
    write_report(output / "fly-code-p3.json", code)
    verification = {"version": 1, "provenance": provenance(ROOT), "sandbox": isolation_status(),
                    "p2_gate": toy["gate"], "p3_gate": code["gate"]}
    if test_log:
        verification["unittest_command"] = "python -m unittest -v"
        verification["unittest_log"] = Path(test_log).read_text()
    write_report(output / "verification.json", verification)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reports")
    parser.add_argument("--test-log", type=Path)
    args = parser.parse_args()
    snapshot(args.run_dir, args.output, test_log=args.test_log)


if __name__ == "__main__":
    main()
