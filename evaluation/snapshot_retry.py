"""Publish compact P6 evidence from a completed, content-locked local experiment."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from evaluation.snapshot_v2 import write_review
from flybrain.graph import digest


ROOT = Path(__file__).resolve().parents[1]


def snapshot(study, destination):
    study, destination = Path(study), Path(destination)
    raw = (study / "report.json").read_bytes()
    report = json.loads(raw)
    started = json.loads((study / "started.json").read_text())
    manifest = json.loads((study / "dataset.json").read_text())
    if (
        report["contract_sha256"] != digest(report["contract"])
        or started["sha256"] != report["contract_sha256"]
        or manifest["sha256"] != digest(manifest["payload"])
        or manifest["sha256"] != report["contract"]["dataset_sha256"]
    ):
        raise ValueError("Retry report/contract/dataset mismatch")
    compact = {k: v for k, v in report.items() if k != "results"}
    compact["artifact_sha256"] = hashlib.sha256(raw).hexdigest()
    compact["recorded_runs"] = len(report["results"])
    compact["failure_shapes_seed_0"] = {
        mode: dict(
            Counter(
                r["shape"]
                for r in report["results"]
                if r["mode"] == mode
                and r["seed"] == 0
                and r["supported"]
                and not r["final_passed"]
            )
        )
        for mode in report["modes"]
    }
    compact["case_results_seed_0"] = [
        {
            "id": r["id"],
            "mode": r["mode"],
            "cohort": r["cohort"],
            "supported": r["supported"],
            "first_passed": r["first_passed"],
            "final_passed": r["final_passed"],
            "status": r["pipeline"]["status"],
            "attempts_used": r["pipeline"]["retry"]["attempts_used"],
        }
        for r in report["results"]
        if r["seed"] == 0
    ]
    compact["examples"] = []
    for mode in report["modes"]:
        row = next(r for r in report["results"] if r["mode"] == mode and r["seed"] == 0)
        attempts = []
        for entry in row["pipeline"]["retry"]["attempts"]:
            actual = entry["result"]
            neural = {}
            for role in ("fly_nl", "fly_code"):
                generation = actual.get(role)
                if generation is None:
                    continue
                events = generation["trace"]
                neural[role] = {
                    "trace_sha256": digest(events),
                    "decisions": len(events),
                    "feedback_sha256": sorted(
                        {
                            e["feedback_sha256"]
                            for e in events
                            if e.get("feedback_sha256")
                        }
                    ),
                    "context_scores": [
                        {k: e[k] for k in ("candidate", "positive_score")}
                        for e in events
                        if e.get("decision") == "context"
                    ],
                    "selected_actions": [
                        e["selected_action"] for e in events if "selected_action" in e
                    ],
                    "first_readout": events[0]["neural"] if events else None,
                }
            attempts.append(
                {
                    "number": entry["number"],
                    "trace_id": actual["trace_id"],
                    "status": actual["status"],
                    "ir": actual["ir"],
                    "code": actual["code"],
                    "failure_feedback": entry.get("failure_feedback"),
                    "input_feedback": entry["input_feedback"],
                    "neural": neural,
                }
            )
        compact["examples"].append(
            {"id": row["id"], "mode": mode, "attempts": attempts}
        )
    lock = ROOT / "experiments/locked/retry-v1.json"
    if lock.exists() and json.loads(lock.read_text()) != manifest:
        raise ValueError("Published retry dataset changed; use a new version")
    destination.mkdir(parents=True, exist_ok=True)
    write_review(destination / "retry-v1.json", compact)
    write_review(lock, manifest)
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "artifacts/retry-v1")
    parser.add_argument("--destination", type=Path, default=ROOT / "docs/reports")
    args = parser.parse_args()
    snapshot(args.study, args.destination)


if __name__ == "__main__":
    main()
