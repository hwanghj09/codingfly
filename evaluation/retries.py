"""Locked, fixed-weight P6 comparison; known v2 phrases, NOT a new blind language test."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

from evaluation.curriculum_v2 import group_cases
from evaluation.datasets import lock_manifest
from evaluation.nl_data import nl_cases, utterances
from evaluation.oracles import grade_code
from evaluation.report import provenance, summary, write_report
from evaluation.sandbox import isolation_status
from flybrain.graph import Graph, digest
from flybrain.retry_input import RETRY_IO_VERSION
from flybrain.sequence_io import lexical_spans
from ir.grammar import lex
from ir.tokens import spec_hash
from models.pipeline import FlyPipeline, load_model
from models.retry import run_retries


ROOT = Path(__file__).resolve().parents[1]
DATA_VERSION = "known-v2-phrase-retry-diagnostics-v1"
PROTOCOL = {
    "io_version": RETRY_IO_VERSION,
    "max_attempts": 3,
    "decoding_seeds": [0, 1, 2, 3, 4],
    "modes": ["resample", "weak", "supervised"],
    "a_retry_context_ranks": [1, 2],
    "b_retry_digit_temperature": 0.3,
    "feedback_input_strength": 0.01,
    "updates": 0,
    "first_attempt": "greedy, no feedback, identical for all arms and seeds",
    "selection": "stop on visible-oracle success, uncertainty or infrastructure failure",
    "hidden_oracle": "full function boundaries, used only after final selection",
    "data_scope": "retrospective v2 test plus fresh literals with the same exposed phrases",
    "supervision": "gold IR only for failed-attempt attribution, never for generation or acceptance",
}


def retry_cases():
    original = deepcopy(nl_cases()["test"])
    for case in original:
        case["cohort"] = "known_v2"
    fresh = []
    shapes = {c["shape"] for c in original if c["supported"]}
    for group in range(300, 304):
        for case in group_cases(group):
            if case["shape"] not in shapes:
                continue
            names = {
                token
                for line in case["source"].splitlines()
                for token in lex(line)
                if token.isidentifier() and token[0].islower()
            }
            raw = utterances(case)[5].encode()
            pieces, previous = [], 0
            for span in lexical_spans(raw):
                word = raw[span.start : span.end].decode()
                pieces.extend(
                    [
                        raw[previous : span.start].decode(),
                        json.dumps(word, ensure_ascii=False) if word in names else word,
                    ]
                )
                previous = span.end
            pieces.append(raw[previous:].decode())
            fresh.append(
                {
                    "id": f"retry-{group}-{case['shape']}",
                    "family": f"retry-name-{group}",
                    "shape": case["shape"],
                    "source": "".join(pieces),
                    "target": case["source"],
                    "expected": case["expected"],
                    "supported": True,
                    "paraphrase": 5,
                    "cohort": "fresh_literals",
                }
            )
        fresh.append(
            {
                "id": f"retry-{group}-unsupported",
                "family": f"retry-name-{group}",
                "shape": "unsupported",
                "source": f"사진 {group} 에 색을 입혀줘",
                "target": "UNKNOWN\n",
                "expected": {},
                "supported": False,
                "paraphrase": 5,
                "cohort": "fresh_literals",
            }
        )
    return {"known_v2": original, "fresh_literals": fresh}


def visible_oracle(expected):
    visible = deepcopy(expected)
    if visible.get("calls"):
        visible["calls"] = visible["calls"][:1]
        visible["returns"] = visible["returns"][:1]
    return visible


def _passed(result, case):
    if not case["supported"]:
        return result["status"] == "uncertain", None
    if result["status"] != "ok":
        return False, None
    hidden = grade_code(result["code"], case["expected"])
    return hidden["semantic_ok"], hidden


def run(code_dir, nl_dir, output):
    output = Path(output)
    if (output / "started.json").exists() or (output / "report.json").exists():
        raise ValueError(
            "Retry study already opened; preserve it and use a new study version"
        )
    isolation = isolation_status()
    if isolation["status"] != "ok":
        raise ValueError(f"Actual OS isolation required: {isolation}")
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    pipeline = FlyPipeline(
        load_model(nl_dir, graph, role="fly-nl"),
        load_model(code_dir, graph, role="fly-code"),
    )
    a_hash, b_hash = (
        pipeline.fly_nl.brain.weights_hash,
        pipeline.fly_code.brain.weights_hash,
    )
    splits = retry_cases()
    manifest = lock_manifest(
        output / "dataset.json",
        splits=splits,
        graph=graph,
        ir_hash=spec_hash(),
        seed=314159,
        kind=DATA_VERSION,
    )
    contract = {
        "protocol": PROTOCOL,
        "dataset_sha256": manifest["sha256"],
        "a_weights_sha256": a_hash,
        "b_weights_sha256": b_hash,
    }
    write_report(
        output / "started.json",
        {"sha256": digest(contract), **contract, "provenance": provenance(ROOT)},
    )
    results, modes = [], {}
    started = time.perf_counter()
    for mode in PROTOCOL["modes"]:
        seeds = []
        for seed in PROTOCOL["decoding_seeds"]:
            rows = []
            for case in [c for cases in splits.values() for c in cases]:
                t0 = time.perf_counter()
                actual = run_retries(
                    pipeline,
                    case["source"],
                    expected=visible_oracle(case["expected"]),
                    expected_ir=case["target"] if mode == "supervised" else None,
                    feedback_mode=mode,
                    seed=seed,
                    record=seed == 0 and case["id"] == splits["known_v2"][0]["id"],
                )
                first = actual["retry"]["attempts"][0]["result"]
                first_pass, first_hidden = _passed(first, case)
                final_pass, final_hidden = _passed(actual, case)
                row = {
                    "id": case["id"],
                    "cohort": case["cohort"],
                    "shape": case["shape"],
                    "supported": case["supported"],
                    "mode": mode,
                    "seed": seed,
                    "first_passed": first_pass,
                    "final_passed": final_pass,
                    "first_hidden": first_hidden,
                    "final_hidden": final_hidden,
                    "seconds": time.perf_counter() - t0,
                    "pipeline": actual,
                }
                rows.append(row)
            positive = [r for r in rows if r["supported"]]
            negative = [r for r in rows if not r["supported"]]
            baseline = sum(r["first_passed"] for r in positive) / len(positive)
            final = sum(r["final_passed"] for r in positive) / len(positive)
            entry = {
                "seed": seed,
                "supported_total": len(positive),
                "pass_at_1": baseline,
                "selected_pass_within_3": final,
                "gain_over_single": final - baseline,
                "unsupported_correct": sum(r["final_passed"] for r in negative),
                "unsupported_total": len(negative),
                "mean_attempts": sum(
                    r["pipeline"]["retry"]["attempts_used"] for r in rows
                )
                / len(rows),
                "seconds": sum(r["seconds"] for r in rows),
                "by_cohort": {
                    cohort: {
                        "supported_total": sum(r["cohort"] == cohort for r in positive),
                        "first_correct": sum(
                            r["first_passed"] and r["cohort"] == cohort
                            for r in positive
                        ),
                        "final_correct": sum(
                            r["final_passed"] and r["cohort"] == cohort
                            for r in positive
                        ),
                    }
                    for cohort in splits
                },
                "final_statuses": dict(Counter(r["pipeline"]["status"] for r in rows)),
            }
            seeds.append(entry)
            results.extend(rows)
            print(mode, entry, flush=True)
        modes[mode] = {
            "seeds": seeds,
            "pass_at_1": summary([s["pass_at_1"] for s in seeds]),
            "selected_pass_within_3": summary(
                [s["selected_pass_within_3"] for s in seeds]
            ),
        }
    if (a_hash, b_hash) != (
        pipeline.fly_nl.brain.weights_hash,
        pipeline.fly_code.brain.weights_hash,
    ):
        raise RuntimeError("Weights changed during retry evaluation")
    result = {
        "phase": "P6-fixed-weight-retry-experiment",
        "provenance": provenance(ROOT),
        "contract": contract,
        "contract_sha256": digest(contract),
        "modes": modes,
        "weights_unchanged": True,
        "learning_updates": 0,
        "seconds": time.perf_counter() - started,
        "interpretation": "Decoding seeds share trained checkpoints; intervals describe search randomness, not training uncertainty. Observed gains are search, not learning.",
        "results": results,
    }
    write_report(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, default=ROOT / "artifacts/code-v2")
    parser.add_argument("--nl-dir", type=Path, default=ROOT / "artifacts/nl-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/retry-v1")
    args = parser.parse_args()
    run(args.code_dir, args.nl_dir, args.output)


if __name__ == "__main__":
    main()
