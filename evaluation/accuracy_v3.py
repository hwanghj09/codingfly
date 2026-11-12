"""One-shot, paired single-attempt evaluation on v3's locked phrase/name test split."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from evaluation.datasets import lock_manifest
from evaluation.nl_data import nl_cases
from evaluation.nl_data_v3 import data_version, nl_cases_v3
from evaluation.oracles import grade_code
from evaluation.report import provenance, write_report
from evaluation.sandbox import isolation_status
from flybrain.graph import Graph
from ir.grammar import InvalidIR, validate
from ir.tokens import spec_hash
from models.fly_nl import FlyNL
from models.pipeline import FlyPipeline, load_model
from models.train_nl import ir_equal, ir_structure, token_similarity
from models.train_nl_v3 import require_unopened


ROOT = Path(__file__).resolve().parents[1]


def require_validation(report):
    validation = report["validation"]
    if (
        validation["supported_semantic_exact_rate"] < 0.95
        or validation["unknown_correct_rate"] < 0.9
    ):
        raise ValueError(
            "Validation gate requires >=95% supported exact IR and >=90% uncertainty; test remains sealed"
        )


def score(pipeline, cases, *, record=False, gold=False):
    results = []
    for case in cases:
        actual = pipeline.run(
            case["source"],
            expected=case["expected"] if case["supported"] else None,
            record=record and not results,
        )
        exact = actual["fly_nl"]["status"] == "eos" and ir_equal(
            actual["ir"], case["target"]
        )
        try:
            validate(actual["ir"])
            valid = True
            structure = (
                ir_structure(actual["ir"]) == ir_structure(case["target"])
                if case["supported"]
                else False
            )
        except InvalidIR:
            valid, structure = False, False
        passed = actual["status"] == ("ok" if case["supported"] else "uncertain")
        gold_result = None
        if gold and case["supported"]:
            generated = pipeline.fly_code.generate(case["target"])
            gold_result = {
                "generation_status": generated.status,
                **grade_code(generated.code, case["expected"]),
            }
            gold_result["passed"] = (
                generated.status == "eos" and gold_result["semantic_ok"]
            )
        failure = "pass" if passed else "A_semantic" if not exact else actual["status"]
        results.append(
            {
                "id": case["id"],
                "shape": case["shape"],
                "supported": case["supported"],
                "passed": passed,
                "ir_exact": exact,
                "ir_valid": valid,
                "ir_structure_exact": structure,
                "failure": failure,
                "token_similarity": token_similarity(actual["ir"], case["target"]),
                "gold_ir_b": gold_result,
                "pipeline": actual,
            }
        )
    positive = [r for r in results if r["supported"]]
    negative = [r for r in results if not r["supported"]]
    return {
        "supported_total": len(positive),
        "correct": sum(r["passed"] for r in positive),
        "pass_at_1": sum(r["passed"] for r in positive) / len(positive),
        "supported_ir_exact_rate": sum(r["ir_exact"] for r in positive) / len(positive),
        "supported_ir_valid_rate": sum(r["ir_valid"] for r in positive) / len(positive),
        "supported_ir_structure_exact_rate": sum(
            r["ir_structure_exact"] for r in positive
        )
        / len(positive),
        "token_similarity": sum(r["token_similarity"] for r in results) / len(results),
        "unsupported_total": len(negative),
        "unsupported_correct": sum(r["passed"] for r in negative),
        "unknown_correct_rate": sum(r["passed"] for r in negative) / len(negative),
        "failure_types": dict(Counter(r["failure"] for r in results)),
        "results": results,
    }


def run(output, code_dir):
    output = Path(output)
    require_unopened(output)
    report = json.loads((output / "report.json").read_text())
    require_validation(report)
    isolation = isolation_status()
    if isolation["status"] != "ok":
        raise ValueError(f"Actual OS isolation required before test: {isolation}")
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    profile = report.get("profile", "initial")
    splits = nl_cases_v3(profile=profile)
    manifest = lock_manifest(
        output / "dataset.json",
        splits=splits,
        graph=graph,
        ir_hash=spec_hash(),
        seed=314159,
        kind=data_version(profile),
    )
    if report["contract"]["dataset_sha256"] != manifest["sha256"]:
        raise ValueError("Report/dataset mismatch")
    selected = load_model(output, graph, role="fly-nl")
    parent = load_model(report["parent_directory"], graph, role="fly-nl")
    b = load_model(code_dir, graph, role="fly-code")
    hashes = [m.brain.weights_hash for m in (parent, selected, b)]
    if hashes[:2] != [report["parent_weights_sha256"], report["final_weights_sha256"]]:
        raise ValueError("Selected or parent checkpoint changed")
    write_report(
        output / "evaluation-started.json",
        {
            "manifest_sha256": manifest["sha256"],
            "model_hashes": hashes,
            "provenance": provenance(ROOT),
            "selection": "single greedy attempt; no retries or feedback",
        },
    )
    baseline = score(FlyPipeline(parent, b), splits["test"])
    hidden = score(FlyPipeline(selected, b), splits["test"], record=True, gold=True)
    regression = score(FlyPipeline(selected, b), nl_cases()["test"])
    frozen = FlyNL(
        graph,
        seed=selected.brain.seed,
        config=selected.brain.config,
        context_bag=selected.encoder.context_bag,
    )
    frozen_exact = sum(
        (g := frozen.generate(c["source"])).status == "eos"
        and ir_equal(g.code, c["target"])
        for c in splits["test"]
    )
    restored = load_model(output, graph, role="fly-nl")
    probe = splits["validation"][0]["source"]
    restore_exact = selected.generate(probe) == restored.generate(probe)
    if hashes != [m.brain.weights_hash for m in (parent, selected, b)]:
        raise RuntimeError("Evaluation modified weights")
    paired = Counter(
        (old["passed"], new["passed"])
        for old, new in zip(baseline["results"], hidden["results"])
        if old["supported"]
    )
    report.update(
        baseline_hidden=baseline,
        hidden=hidden,
        known_v2_regression=regression,
        frozen_ir_exact={"correct": frozen_exact, "total": len(splits["test"])},
        changed_edges_from_parent=int(
            np.count_nonzero(selected.brain.weights != parent.brain.weights)
        ),
        b_weights_sha256=hashes[2],
        weights_unchanged_during_evaluation=True,
        restore_exact=restore_exact,
        test_evaluated=True,
        provenance=provenance(ROOT),
        paired_supported={
            "both_correct": paired[True, True],
            "improved": paired[False, True],
            "regressed": paired[True, False],
            "both_wrong": paired[False, False],
        },
    )
    write_report(output / "hidden.json", report)
    write_report(output / "report.json", report)
    print(
        {
            "baseline": baseline["pass_at_1"],
            "selected": hidden["pass_at_1"],
            "unknown": hidden["unknown_correct_rate"],
            "old_phrases": regression["pass_at_1"],
            "paired": report["paired_supported"],
        },
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/nl-v3")
    parser.add_argument("--code-dir", type=Path, default=ROOT / "artifacts/code-v2")
    args = parser.parse_args()
    run(args.output, args.code_dir)


if __name__ == "__main__":
    main()
