"""Fixed-model A->B diagnostics; requires that NL hidden evaluation was already sealed."""

import argparse
from collections import Counter
import json
from pathlib import Path

from evaluation.nl_data import nl_cases
from evaluation.oracles import grade_code
from evaluation.report import provenance, write_report
from flybrain.graph import Graph
from models.pipeline import FlyPipeline, load_model
from models.train_nl import ir_equal


ROOT = Path(__file__).resolve().parents[1]


def run(code_dir, nl_dir, output):
    report = json.loads((Path(nl_dir) / "report.json").read_text())
    if not report.get("test_evaluated"):
        raise ValueError("Seal the NL hidden evaluation before end-to-end diagnostics")
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    a = load_model(nl_dir, graph, role="fly-nl")
    b = load_model(code_dir, graph, role="fly-code")
    pipeline = FlyPipeline(a, b)
    a_hash, b_hash = a.brain.weights_hash, b.brain.weights_hash
    results = []
    for case in nl_cases()["test"]:
        actual = pipeline.run(
            case["source"],
            expected=case["expected"] if case["supported"] else None,
            record=not results,
        )
        gold = None
        if case["supported"]:
            generation = b.generate(case["target"])
            gold = {
                **grade_code(generation.code, case["expected"]),
                "generation_status": generation.status,
            }
            gold["passed"] = generation.status == "eos" and gold["semantic_ok"]
        passed = actual["status"] == ("ok" if case["supported"] else "uncertain")
        exact = ir_equal(actual["ir"], case["target"])
        failure = (
            "pass"
            if passed
            else "A_unsupported"
            if not case["supported"]
            else "A_generation"
            if actual["status"] == "nl_generation_error"
            else "A_ir_format"
            if actual["status"] == "invalid_ir"
            else "A_semantic"
            if not exact
            else "B_generation_or_execution"
        )
        results.append(
            {
                "id": case["id"],
                "shape": case["shape"],
                "supported": case["supported"],
                "target_ir": case["target"],
                "ir_exact": exact,
                "passed": passed,
                "failure": failure,
                "gold_ir_b": gold,
                "pipeline": actual,
            }
        )
    supported = [r for r in results if r["supported"]]
    unknown = [r for r in results if not r["supported"]]
    assert a.brain.weights_hash == a_hash and b.brain.weights_hash == b_hash, (
        "Evaluation must never learn hidden cases"
    )
    result = {
        "phase": "P5-experiment",
        "provenance": provenance(ROOT),
        "graph_sha256": graph.topology_hash,
        "a_weights_sha256": a_hash,
        "b_weights_sha256": b_hash,
        "supported_total": len(supported),
        "e2e_pass_at_1": sum(r["passed"] for r in supported) / len(supported),
        "gold_ir_b_pass_at_1": sum(r["gold_ir_b"]["passed"] for r in supported)
        / len(supported),
        "unsupported_total": len(unknown),
        "unsupported_correct_rate": sum(r["passed"] for r in unknown) / len(unknown),
        "failure_types": dict(Counter(r["failure"] for r in results)),
        "unique_trace_ids": len({r["pipeline"]["trace_id"] for r in results}),
        "weights_unchanged_during_evaluation": True,
        "results": results,
    }
    write_report(Path(output) / "report.json", result)
    print(
        {k: v for k, v in result.items() if k not in {"results", "provenance"}},
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, default=ROOT / "artifacts/code-v2")
    parser.add_argument("--nl-dir", type=Path, default=ROOT / "artifacts/nl-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/e2e-v2")
    args = parser.parse_args()
    run(args.code_dir, args.nl_dir, args.output)


if __name__ == "__main__":
    main()
