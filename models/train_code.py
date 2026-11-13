"""Fixed-budget P3 experiment, unlocked only by a successful five-seed P2 report."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np

from evaluation.code_data import code_cases
from evaluation.datasets import lock_manifest
from evaluation.oracles import grade_code
from evaluation.report import provenance, write_report
from flybrain.graph import Graph
from flybrain.decoder import whitespace_spans
from flybrain.policy import verify_generation_tree
from ir.tokens import IR_VERSION, spec_hash
from models.fly_code import FlyCode


ROOT = Path(__file__).resolve().parents[1]


def require_p2(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("phase") != "P2" or report.get("gate", {}).get("passed") is not True:
        raise ValueError("Roadmap P2 gate has not passed; stop before P3")
    learned = {r["seed"]: r for r in report["runs"] if r["condition"] == "learned"}
    frozen = {r["seed"]: r for r in report["runs"] if r["condition"] == "frozen"}
    if len(learned) < 5 or learned.keys() != frozen.keys():
        raise ValueError("P2 requires at least five paired independent seeds")
    for seed, current in learned.items():
        control = frozen[seed]
        if not (current["test_accuracy"] >= .75 and current["test_accuracy"] > control["test_accuracy"]
                and current["changed_edges"] > 0 and control["changed_edges"] == 0
                and current["restore_exact"] and control["restore_exact"]
                and current["cut_test_accuracy"] <= .25
                and current["initial_weights_sha256"] == control["initial_weights_sha256"]
                and current["final_weights_sha256"] != current["initial_weights_sha256"]
                and control["final_weights_sha256"] == control["initial_weights_sha256"]):
            raise ValueError(f"P2 evidence failed at seed {seed}")
    return {"dataset_sha256": report["dataset_sha256"], "graph_sha256": report["graph"]["graph_sha256"],
            "seeds": sorted(learned)}


def evaluate(model, cases, *, record=False):
    results = []
    for case in cases:
        generation = model.generate(case["source"], max_actions=32, record=record and not results)
        grade = grade_code(generation.code, case["expected"])
        # A truncated prefix that happens to run does not count as completion.
        passed = generation.status == "eos" and grade["semantic_ok"]
        results.append({"id": case["id"], "source": case["source"], "shape": case["shape"],
                        "expected_code": case["target"], "generation": asdict(generation),
                        "exact_match": generation.code == case["target"],
                        "grade": grade, "passed": passed})
    total = len(results)
    syntax_count = sum(result["grade"]["syntax_ok"] for result in results)
    return {"total": total, "syntax_rate": syntax_count / total,
            "semantic_rate_of_syntactically_valid": sum(r["passed"] for r in results) / syntax_count
            if syntax_count else 0., "pass_at_1": sum(result["passed"] for result in results) / total,
            "exact_match_rate": sum(result["exact_match"] for result in results) / total,
            "failure_types": dict(Counter(result["generation"]["status"] if
                                           result["generation"]["status"] != "eos" else
                                           result["grade"]["status"] if not result["passed"] else "pass"
                                           for result in results)), "results": results}


def run(output, toy_report, *, graph_path=ROOT / "flywire_lab_2048.csv", seed=0, epochs=40,
        learning_rate=.03):
    if epochs < 1:
        raise ValueError("Positive epoch budget required")
    policy_files = verify_generation_tree()
    gate = require_p2(toy_report)
    graph, output = Graph.from_csv(graph_path), Path(output)
    splits = code_cases()
    manifest = lock_manifest(output / "dataset.json", splits=splits, graph=graph, ir_hash=spec_hash(),
                             seed=1729, kind="P3-LET-PRINT-surface-holdout-v1")
    contract = {"dataset_sha256": manifest["sha256"], "ir_version": IR_VERSION, "ir_sha256": spec_hash()}
    records = []
    for condition in ("learned", "frozen"):
        model = FlyCode(graph, seed=seed)
        initial_hash = model.brain.weights_hash
        examples = [example for case in splits["train"]
                    for example in model.teacher_examples(case["source"], case["actions"])]
        # Ensure the teacher data is internally consistent with the fixed decoder.
        for case in splits["train"]:
            source = case["source"].encode()
            rendered = b"".join(model.decoder.emit(a, source, whitespace_spans(source)) for a in case["actions"])
            if rendered.decode() != case["target"]:
                raise ValueError("Teacher action/code mismatch")
        order, rng, history = list(range(len(examples))), random.Random(seed), []
        started = time.perf_counter()
        for epoch in range(epochs):
            rng.shuffle(order)
            losses, correct = [], 0
            for i in order:
                stimulus, expected = examples[i]
                result = model.train_example(stimulus, expected, learning_rate=learning_rate,
                                             frozen=condition == "frozen")
                losses.append(result["loss"])
                correct += result["chosen"] == expected
            history.append({"epoch": epoch + 1, "teacher_loss": float(np.mean(losses)),
                            "teacher_action_accuracy": correct / len(examples)})
        seconds = time.perf_counter() - started
        model.save(output / f"{condition}.json", contract=contract,
                   training_state={"epochs": epochs, "shuffle_rng_state": rng.getstate()})
        restored, _ = FlyCode.restore(output / f"{condition}.json", graph, contract=contract)
        probe = splits["validation"][0]["source"]
        restore_exact = model.generate(probe) == restored.generate(probe)
        # Evaluation happens only after the fixed training budget. No test-driven retries.
        validation = evaluate(model, splits["validation"])
        hidden = evaluate(model, splits["test"], record=True)
        print(f"{condition}: teacher={history[-1]['teacher_action_accuracy']:.3f}, "
              f"hidden syntax={hidden['syntax_rate']:.3f}, pass@1={hidden['pass_at_1']:.3f}", flush=True)
        records.append({"condition": condition, "seed": seed, "initial_weights_sha256": initial_hash,
                        "final_weights_sha256": model.brain.weights_hash, "updates": model.brain.updates,
                        "changed_edges": int(np.count_nonzero(model.brain.weights != model.brain.initial_weights)),
                        "restore_exact": restore_exact, "training_seconds": seconds,
                        "history": history, "validation": validation, "hidden": hidden})
    hidden = records[0]["hidden"]
    targets_met = hidden["syntax_rate"] >= .9 and hidden["semantic_rate_of_syntactically_valid"] >= .8
    # LET/PRINT surface holdout cannot satisfy the roadmap's larger control-flow/algorithm scope.
    report = {"version": 1, "phase": "P3-experiment", "provenance": provenance(ROOT),
              "p2_gate": gate, "graph": graph.metadata(), "contract": contract,
              "policy_files": policy_files, "epochs": epochs, "learning_rate": learning_rate,
              "runs": records, "gate": {"passed": False, "numeric_targets_met": targets_met,
                                         "next": "STOP before P4: complete Fly-Code generalization and control flow"},
              "limits": ["Only LET, PRINT and their composition; no IF/FOR/FUNC yet",
                         "24 hidden programs in 8 name/value groups; not 24 algorithm families",
                         "Single seed for P3; five-seed evidence currently applies to P2 only",
                         "No Fly-NL, end-to-end generation or fallback; invalid predictions are retained"]}
    write_report(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/fly-code")
    parser.add_argument("--toy-report", type=Path, default=ROOT / "artifacts/toy-p2/report.json")
    parser.add_argument("--graph", type=Path, default=ROOT / "flywire_lab_2048.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=.03)
    args = parser.parse_args()
    run(args.output, args.toy_report, graph_path=args.graph, seed=args.seed,
        epochs=args.epochs, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
