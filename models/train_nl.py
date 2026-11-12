"""P4 training is gated by measured P3 performance; no regex or model fallback."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from evaluation.curriculum_v2 import teacher_actions
from evaluation.datasets import lock_manifest
from evaluation.nl_data import NL_DATA_VERSION, nl_cases
from evaluation.report import provenance, write_report
from flybrain.graph import Graph
from ir.grammar import InvalidIR, lex, validate
from ir.serializer import canonicalize
from ir.tokens import IR_VERSION, spec_hash, COMMANDS, BINARY, UNARY
from models.fly_nl import FlyNL, NL_VOCABULARY
from models.neural_sequence import NeuralSequence
from models.train_sequence import fit


ROOT = Path(__file__).resolve().parents[1]


def require_p3(path):
    report = json.loads(Path(path).read_text())
    hidden = report.get("hidden", {})
    if (
        report.get("phase") != "P3-v2"
        or not report.get("test_evaluated")
        or not report.get("gate", {}).get("passed")
        or hidden.get("shapes", 0) < 20
        or hidden.get("syntax_rate", 0) < 0.9
        or hidden.get("semantic_rate_of_syntactically_valid", 0) < 0.8
    ):
        raise ValueError(
            "P3 execution/generalization gate has not passed; stop before P4"
        )
    return {
        "contract": report["contract"],
        "final_weights_sha256": report["final_weights_sha256"],
    }


def nl_teacher(source, target, vocabulary):
    if "\n" in source:
        raise ValueError("P4 curriculum expects a single utterance line")
    body = teacher_actions(source, target, vocabulary)[:-1]
    commands = tuple(line.split()[0] for line in target.splitlines())
    contexts = {
        ("PRINT",): 0,
        ("LET",): 1,
        ("LET", "PRINT"): 2,
        ("LET", "ADD", "PRINT"): 3,
        ("FOR", "PRINT", "END"): 6,
        ("UNKNOWN",): 7,
    }
    if commands == ("FUNC", "RETURN", "END_FUNC"):
        label = 5 if "(" in target.splitlines()[1] else 4
    else:
        label = contexts[commands]
    pointers = []
    for action in body:
        if (
            len(vocabulary.literals) <= action < vocabulary.next_line
            and action not in pointers
        ):
            pointers.append(action)
    header = [vocabulary.context_start + label]
    for pointer in pointers:
        header.extend([vocabulary.bind, pointer])
    header.append(vocabulary.begin_text)
    text = [
        vocabulary.register_start + pointers.index(action)
        if action in pointers
        else action
        for action in body
    ]
    return header + text + [vocabulary.next_line, 0]


def ir_equal(predicted, expected):
    if expected.strip() == "UNKNOWN":
        return predicted.strip() == "UNKNOWN"
    try:
        return canonicalize(predicted) == canonicalize(expected)
    except InvalidIR:
        return False


def ir_tokens(text):
    try:
        return [token for line in text.splitlines() for token in [*lex(line), "<EOL>"]]
    except InvalidIR:
        return list(text)


def token_similarity(predicted, expected):
    first, second = ir_tokens(predicted), ir_tokens(expected)
    row = list(range(len(second) + 1))
    for i, one in enumerate(first, 1):
        next_row = [i]
        for j, two in enumerate(second, 1):
            next_row.append(
                min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (one != two))
            )
        row = next_row
    return max(0.0, 1 - row[-1] / max(1, len(first), len(second)))


def ir_structure(text):
    reserved = COMMANDS | BINARY | UNARY | {"RANGE", "LIST", "TRUE", "FALSE", "(", ")"}
    return tuple(
        (
            statement.command,
            tuple(
                token
                if token in reserved
                else "STRING"
                if token.startswith('"')
                else "NAME"
                if token.isidentifier()
                else "INTEGER"
                for token in statement.tokens
            ),
        )
        for statement in validate(text)
    )


def nl_validation(model, cases):
    return sum(
        (g := model.generate(c["source"])).status == "eos"
        and ir_equal(g.code, c["target"])
        for c in cases
    ) / len(cases)


def evaluate_nl(model, cases, *, trace=False):
    results = []
    for case in cases:
        generated = model.generate(case["source"], record=trace and not results)
        uncertain = generated.code.strip() == "UNKNOWN"
        try:
            canonicalize(generated.code)
            valid = True
            structural = (
                ir_structure(generated.code) == ir_structure(case["target"])
                if case["supported"]
                else False
            )
        except InvalidIR:
            valid, structural = False, False
        semantic = generated.status == "eos" and ir_equal(
            generated.code, case["target"]
        )
        results.append(
            {
                "id": case["id"],
                "shape": case["shape"],
                "source": case["source"],
                "target": case["target"],
                "generation": asdict(generated),
                "ir_valid": valid,
                "uncertain": uncertain,
                "expected_supported": case["supported"],
                "semantic_exact": semantic,
                "structure_exact": structural,
                "token_similarity": token_similarity(generated.code, case["target"]),
            }
        )
    supported = [r for r in results if r["expected_supported"]]
    unknown = [r for r in results if not r["expected_supported"]]
    return {
        "total": len(results),
        "token_similarity": sum(r["token_similarity"] for r in results) / len(results),
        "supported_ir_valid_rate": sum(r["ir_valid"] for r in supported)
        / len(supported),
        "supported_structure_exact_rate": sum(r["structure_exact"] for r in supported)
        / len(supported),
        "supported_semantic_exact_rate": sum(r["semantic_exact"] for r in supported)
        / len(supported),
        "unknown_correct_rate": sum(r["semantic_exact"] for r in unknown)
        / len(unknown),
        "overall_semantic_exact_rate": sum(r["semantic_exact"] for r in results)
        / len(results),
        "results": results,
    }


def open_nl_hidden(output):
    output = Path(output)
    report = json.loads((output / "report.json").read_text())
    if report.get("test_evaluated") or (output / "hidden.json").exists():
        raise ValueError("NL hidden evaluation already opened; do not tune and reopen")
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    best, _ = NeuralSequence.restore(
        output / "best.json",
        graph,
        NL_VOCABULARY,
        role="fly-nl",
        contract=report["contract"],
    )
    splits = nl_cases()
    report.update(
        hidden=evaluate_nl(best, splits["test"], trace=True),
        frozen=evaluate_nl(FlyNL(graph, seed=report["seed"]), splits["test"]),
        test_evaluated=True,
        provenance=provenance(ROOT),
    )
    restored, _ = NeuralSequence.restore(
        output / "best.json",
        graph,
        NL_VOCABULARY,
        role="fly-nl",
        contract=report["contract"],
    )
    probe = splits["validation"][0]["source"]
    report["restore_exact"] = best.generate(probe) == restored.generate(probe)
    write_report(output / "hidden.json", report)
    write_report(output / "report.json", report)
    print(
        "NL hidden",
        {key: value for key, value in report["hidden"].items() if key != "results"},
        flush=True,
    )
    return report


def run(
    output, p3_report, *, epochs=40, seed=101, evaluate_test=False, polish_steps=160
):
    p3 = require_p3(p3_report)
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    output = Path(output)
    if (output / "hidden.json").exists():
        raise ValueError("Cannot train after opening this NL study hidden test")
    splits = nl_cases()
    manifest = lock_manifest(
        output / "dataset.json",
        splits=splits,
        graph=graph,
        ir_hash=spec_hash(),
        seed=314159,
        kind=NL_DATA_VERSION,
    )
    contract = {
        "dataset_sha256": manifest["sha256"],
        "ir_version": IR_VERSION,
        "ir_sha256": spec_hash(),
    }
    model = FlyNL(graph, seed=seed)
    initial = model.brain.weights_hash
    training = fit(
        model,
        splits,
        output,
        contract,
        epochs=epochs,
        action_builder=nl_teacher,
        validation_scorer=nl_validation,
    )
    best, state = NeuralSequence.restore(
        output / "best.json", graph, NL_VOCABULARY, role="fly-nl", contract=contract
    )
    report = {
        "phase": "P4-experiment",
        "provenance": provenance(ROOT),
        "p3_gate": p3,
        "contract": contract,
        "seed": seed,
        "initial_weights_sha256": initial,
        "final_weights_sha256": best.brain.weights_hash,
        "changed_edges": int(
            np.count_nonzero(best.brain.weights != best.brain.initial_weights)
        ),
        "training": training,
        "selected_epoch": state["epoch"],
        "validation": evaluate_nl(best, splits["validation"]),
        "test_evaluated": False,
        "limits": [
            "Eight controlled intents plus uncertainty; not arbitrary-language understanding",
            "Paraphrase forms 4/5 and name groups are disjoint from training forms 0..3",
        ],
    }
    write_report(output / "report.json", report)
    if polish_steps:
        from models.train_sequence import polish_saved

        report = polish_saved(output, role="fly-nl", steps=polish_steps)
    if evaluate_test:
        report = open_nl_hidden(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/nl-v2")
    parser.add_argument(
        "--p3-report", type=Path, default=ROOT / "artifacts/code-v2/report.json"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--polish-steps", type=int, default=160)
    args = parser.parse_args()
    run(
        args.output,
        args.p3_report,
        epochs=args.epochs,
        seed=args.seed,
        evaluate_test=args.evaluate_test,
        polish_steps=args.polish_steps,
    )


if __name__ == "__main__":
    main()
