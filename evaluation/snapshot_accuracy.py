"""Compact evidence for a completed v3 accuracy study; full traces remain local."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from evaluation.snapshot_v2 import model_metadata, write_review
from flybrain.clock_encoder import ClockEncoder
from flybrain.graph import digest
from flybrain.graph import Graph
from flybrain.sequence_state import SequenceState
from models.pipeline import load_model


ROOT = Path(__file__).resolve().parents[1]


def snapshot(directory, destination):
    directory, destination = Path(directory), Path(destination)
    raw = (directory / "report.json").read_bytes()
    report = json.loads(raw)
    manifest = json.loads((directory / "dataset.json").read_text())
    if (
        not report.get("test_evaluated")
        or manifest["sha256"] != digest(manifest["payload"])
        or manifest["sha256"] != report["contract"]["dataset_sha256"]
    ):
        raise ValueError("Completed, matching v3 study required")
    lock = ROOT / "experiments/locked/nl-v3.json"
    if lock.exists() and json.loads(lock.read_text()) != manifest:
        raise ValueError("Published v3 dataset changed; use a new dataset version")
    result = deepcopy(report)
    result["full_report_sha256"] = hashlib.sha256(raw).hexdigest()
    result["model"] = model_metadata(directory)
    splits = manifest["payload"]["splits"]
    masked = {
        name: {
            hashlib.sha256(
                ClockEncoder.features(c["source"].encode(), False, True)[0].tobytes()
            ).hexdigest()
            for c in rows
        }
        for name, rows in splits.items()
    }
    result["dataset_audit"] = {
        "counts": {
            name: {
                "total": len(rows),
                "supported": sum(c["supported"] for c in rows),
                "masked_phrase_types": len(masked[name]),
            }
            for name, rows in splits.items()
        },
        "literal_masked_ordered_phrase_overlap": {
            f"{first}/{second}": len(masked[first] & masked[second])
            for first, second in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        },
    }
    model = load_model(
        directory, Graph.from_csv(ROOT / "flywire_lab_2048.csv"), role="fly-nl"
    )
    context_hashes = {
        name: [
            hashlib.sha256(
                SequenceState(c["source"].encode().splitlines(), model.vocabulary)
                .context_stimulus(model.encoder, 0)
                .tobytes()
            ).hexdigest()
            for c in rows
        ]
        for name, rows in splits.items()
    }
    train_inputs = set(context_hashes["train"])
    result["dataset_audit"]["context_view"] = {
        "meaning": "Word-order-normalized context stimuli can coincide across phrase splits; pointers and full utterances remain separately evaluated",
        "distinct_inputs": {
            name: len(set(values)) for name, values in context_hashes.items()
        },
        "test_cases_with_context_input_seen_in_train": {
            "supported": sum(
                value in train_inputs and case["supported"]
                for value, case in zip(context_hashes["test"], splits["test"])
            ),
            "unsupported": sum(
                value in train_inputs and not case["supported"]
                for value, case in zip(context_hashes["test"], splits["test"])
            ),
        },
    }
    cases = {
        case["id"]: case
        for split in manifest["payload"]["splits"].values()
        for case in split
    }
    for section in ("baseline_validation", "warm_start_validation", "validation"):
        if section not in result:
            continue
        result[section] = {k: v for k, v in result[section].items() if k != "results"}
    for section in ("baseline_hidden", "hidden", "known_v2_regression"):
        for row in result[section]["results"]:
            actual = row.pop("pipeline")
            row.update(
                status=actual["status"],
                ir=actual["ir"],
                code=actual["code"],
                trace_id=actual["trace_id"],
                verification=actual.get("verification"),
            )
            if row["id"] in cases:
                row.update(
                    source=cases[row["id"]]["source"],
                    target_ir=cases[row["id"]]["target"],
                )
    first = report["hidden"]["results"][0]["pipeline"]
    result["neural_evidence"] = {}
    for role in ("fly_nl", "fly_code"):
        generation = first.get(role)
        if generation:
            events = generation["trace"]
            result["neural_evidence"][role] = {
                "trace_sha256": digest(events),
                "decisions": len(events),
                "trace_sample": events[:12],
                "actions": generation["actions"],
            }
    write_review(lock, manifest)
    write_review(destination / "accuracy-v3.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/nl-v3")
    parser.add_argument("--destination", type=Path, default=ROOT / "docs/reports")
    args = parser.parse_args()
    snapshot(args.directory, args.destination)


if __name__ == "__main__":
    main()
