"""Train neural sequences on train only; select on validation; test is an explicit one-shot step."""

import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from evaluation.curriculum_v2 import (
    DATA_VERSION,
    aligned_code_actions,
    code_cases_v2,
    composition_cases,
)
from evaluation.datasets import lock_manifest
from evaluation.oracles import grade_code
from evaluation.report import provenance, write_report
from flybrain.batch import EdgeAdam, loss_gradient
from flybrain.graph import Graph
from ir.tokens import IR_VERSION, spec_hash
from models.fly_code_v2 import CODE_VOCABULARY, FlyCodeV2
from models.neural_sequence import NeuralSequence
from models.train_code import require_p2


ROOT = Path(__file__).resolve().parents[1]


def encode_training(model, cases, output, action_builder=aligned_code_actions):
    actions = [
        action_builder(c["source"], c["target"], model.vocabulary) for c in cases
    ]
    count = sum(
        model.teacher_decision_count(case["source"], sequence)
        for case, sequence in zip(cases, actions)
    )
    matrix = np.lib.format.open_memmap(
        output / "train-stimuli.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, len(model.brain.state)),
    )
    targets = np.zeros(count, dtype=np.int64)
    position = 0
    seen = {}
    for index, (case, sequence) in enumerate(zip(cases, actions)):
        for stimulus, target in model.teacher_decisions(case["source"], sequence):
            key = hashlib.sha256(stimulus.tobytes()).digest()
            if key in seen:
                if targets[seen[key]] != target:
                    raise ValueError(
                        "Indistinguishable training inputs have conflicting teacher actions"
                    )
                continue
            seen[key] = position
            matrix[position], targets[position] = stimulus, target
            position += 1
        if (index + 1) % 48 == 0:
            print(f"encoded {index + 1}/{len(cases)} training programs", flush=True)
    matrix.flush()
    print(
        f"{count} teacher observations -> {position} distinct fixed input states",
        flush=True,
    )
    return matrix[:position], targets[:position]


def exact_validation(model, cases):
    results = []
    for case in cases:
        generated = model.generate(case["source"], max_actions=512)
        results.append(generated.status == "eos" and generated.code == case["target"])
    return sum(results) / len(results)


def evaluate_code(model, cases, *, trace=False):
    results = []
    for case in cases:
        generated = model.generate(
            case["source"], max_actions=512, record=trace and not results
        )
        grade = grade_code(generated.code, case["expected"])
        results.append(
            {
                "id": case["id"],
                "shape": case["shape"],
                "source": case["source"],
                "target": case["target"],
                "generation": asdict(generated),
                "grade": grade,
                "passed": generated.status == "eos" and grade["semantic_ok"],
            }
        )
    valid = sum(r["grade"]["syntax_ok"] for r in results)
    passed = sum(r["passed"] for r in results)
    return {
        "total": len(results),
        "syntax_rate": valid / len(results),
        "pass_at_1": passed / len(results),
        "semantic_rate_of_syntactically_valid": passed / valid if valid else 0.0,
        "shapes": len({r["shape"] for r in results}),
        "results": results,
    }


def polish(model, matrix, targets, *, steps=120):
    """Bounded full-batch BPTT optimization, still only original edge coefficients."""
    if type(steps) is not int or steps < 1:
        raise ValueError("Positive polishing budget required")
    brain = model.brain
    bounds = (
        [(-brain.config.max_weight, brain.config.max_weight)] * len(brain.weights)
        if not brain.config.sign_constraint
        else [
            (0.0, brain.config.max_weight)
            if sign > 0
            else (-brain.config.max_weight, 0.0)
            for sign in brain.signs
        ]
    )
    iterations = 0
    started = time.perf_counter()

    def statistics():
        if len(targets) <= 4096:
            return loss_gradient(brain, matrix, targets)
        loss_sum, gradient, probabilities = 0.0, np.zeros_like(brain.weights), []
        for start in range(0, len(targets), 512):
            batch = targets[start : start + 512]
            loss, derivative, prediction = loss_gradient(
                brain, matrix[start : start + 512], batch
            )
            loss_sum += loss * len(batch)
            gradient += derivative * len(batch)
            probabilities.append(prediction)
        return (
            loss_sum / len(targets),
            gradient / len(targets),
            np.concatenate(probabilities),
        )

    def objective(weights):
        brain.weights = weights.copy()
        loss, gradient, _ = statistics()
        brain.eligibility = -gradient
        return loss, gradient

    def progress(weights):
        nonlocal iterations
        iterations += 1
        brain.updates += 1
        brain.steps_seen += len(targets) * brain.config.steps
        if iterations % 20 == 0:
            print(f"L-BFGS iteration {iterations}", flush=True)

    result = minimize(
        objective,
        brain.weights.copy(),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        callback=progress,
        options={"maxiter": steps, "ftol": 1e-11, "gtol": 1e-7, "maxls": 30},
    )
    brain.weights = result.x.copy()
    loss, _, probabilities = statistics()
    return {
        "method": "bounded L-BFGS-B on existing edges",
        "iterations": int(result.nit),
        "loss": loss,
        "teacher_digit_accuracy": float(
            np.mean(probabilities.argmax(axis=1) == targets)
        ),
        "seconds": time.perf_counter() - started,
        "termination": str(result.message),
    }


def polish_saved(output, *, steps=120, role="fly-code", checkpoint="best.json"):
    output = Path(output)
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    report = json.loads((output / "report.json").read_text())
    if report.get("test_evaluated") or (output / "hidden.json").exists():
        raise ValueError("Cannot tune a model after opening its hidden test")
    if role == "fly-nl":
        from evaluation.nl_data import nl_cases
        from models.fly_nl import NL_VOCABULARY
        from models.train_nl import nl_teacher, evaluate_nl

        vocabulary, splits, builder, evaluator = (
            NL_VOCABULARY,
            nl_cases(),
            nl_teacher,
            evaluate_nl,
        )
    else:
        vocabulary, splits, builder, evaluator = (
            CODE_VOCABULARY,
            code_cases_v2(),
            aligned_code_actions,
            evaluate_code,
        )
    manifest = json.loads((output / "dataset.json").read_text())
    contract = {
        "dataset_sha256": manifest["sha256"],
        "ir_version": IR_VERSION,
        "ir_sha256": spec_hash(),
    }
    model, state = NeuralSequence.restore(
        output / checkpoint, graph, vocabulary, role=role, contract=contract
    )
    matrix, targets = encode_training(model, splits["train"], output, builder)
    polishing = polish(model, matrix, targets, steps=steps)
    validation = evaluator(model, splits["validation"])
    key = "pass_at_1" if role == "fly-code" else "overall_semantic_exact_rate"
    accepted = validation[key] >= report["validation"][key]
    if accepted:
        model.save(
            output / "best.json",
            contract=contract,
            training_state={
                "epoch": state.get("epoch"),
                "optimizer": "L-BFGS-B",
                "polishing": polishing,
            },
        )
    else:
        model, _ = NeuralSequence.restore(
            output / "best.json", graph, vocabulary, role=role, contract=contract
        )
        validation = report["validation"]
    report.update(
        provenance=provenance(ROOT),
        final_weights_sha256=model.brain.weights_hash,
        changed_edges=int(
            np.count_nonzero(model.brain.weights != model.brain.initial_weights)
        ),
        validation=validation,
        selected_stage="L-BFGS-B" if accepted else "Adam",
    )
    polishing["accepted_by_validation"] = accepted
    report["training"]["polishing"] = polishing
    write_report(output / "report.json", report)
    print("polished validation", report["validation"][key], flush=True)
    return report


def open_hidden(output):
    """Freeze extra compositions, then evaluate the selected checkpoint exactly once."""
    output = Path(output)
    report = json.loads((output / "report.json").read_text())
    if report.get("test_evaluated") or (output / "hidden.json").exists():
        raise ValueError(
            "This study already opened hidden evaluation; use a new dataset version for further tuning"
        )
    validation = report["validation"]
    if validation["syntax_rate"] < 0.9 or validation["pass_at_1"] < 0.8:
        raise ValueError("Validation target failed; hidden data stays sealed")
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    best, _ = NeuralSequence.restore(
        output / "best.json",
        graph,
        CODE_VOCABULARY,
        role="fly-code",
        contract=report["contract"],
    )
    cases = composition_cases()
    composition_manifest = lock_manifest(
        output / "composition.json",
        splits={"composition": cases},
        graph=graph,
        ir_hash=spec_hash(),
        seed=314159,
        kind="held-out-compositions-v2",
    )
    splits = code_cases_v2()
    hidden = evaluate_code(best, splits["test"], trace=True)
    compositions = evaluate_code(best, cases)
    control = evaluate_code(FlyCodeV2(graph, seed=report["seed"]), splits["test"])
    restored, _ = NeuralSequence.restore(
        output / "best.json",
        graph,
        CODE_VOCABULARY,
        role="fly-code",
        contract=report["contract"],
    )
    probe = splits["validation"][0]["source"]
    report.update(
        hidden=hidden,
        composition=compositions,
        frozen=control,
        test_evaluated=True,
        composition_manifest_sha256=composition_manifest["sha256"],
        restore_exact=best.generate(probe) == restored.generate(probe),
        provenance=provenance(ROOT),
    )
    report["gate"] = {
        "passed": hidden["shapes"] >= 20
        and hidden["syntax_rate"] >= 0.9
        and hidden["semantic_rate_of_syntactically_valid"] >= 0.8
        and compositions["pass_at_1"] >= 0.75,
        "criteria": "20+ controlled types; syntax >= .90; semantic >= .80; new compositions >= .75",
    }
    write_report(output / "hidden.json", report)
    write_report(output / "report.json", report)
    print(
        "hidden",
        hidden["syntax_rate"],
        hidden["pass_at_1"],
        "compositions",
        compositions["pass_at_1"],
        report["gate"],
        flush=True,
    )
    return report


def fit(
    model,
    splits,
    output,
    contract,
    *,
    epochs=40,
    batch_size=64,
    learning_rate=0.01,
    action_builder=aligned_code_actions,
    validation_scorer=exact_validation,
):
    output = Path(output)
    if (output / "hidden.json").exists():
        raise ValueError("Cannot train after opening this study hidden test")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Positive epoch and batch budgets required")
    output.mkdir(parents=True, exist_ok=True)
    matrix, targets = encode_training(model, splits["train"], output, action_builder)
    optimizer = EdgeAdam(model.brain, learning_rate=learning_rate)
    rng = np.random.default_rng(model.brain.seed)
    history, best, best_loss = [], -1.0, float("inf")
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(targets))
        loss, correct = 0.0, 0
        for start in range(0, len(targets), batch_size):
            index = order[start : start + batch_size]
            result = optimizer.update(matrix[index], targets[index])
            loss += result["loss"] * len(index)
            correct += result["correct"]
        record = {
            "epoch": epoch,
            "teacher_loss": loss / len(targets),
            "teacher_digit_accuracy": correct / len(targets),
        }
        if epoch % 10 == 0 or epoch == epochs:
            record["validation_exact"] = validation_scorer(model, splits["validation"])
            state = {
                "epoch": epoch,
                "optimizer": optimizer.state(),
                "rng": rng.bit_generator.state,
            }
            model.save(output / "latest.json", contract=contract, training_state=state)
            if (
                record["validation_exact"] > best
                or record["validation_exact"] == best
                and record["teacher_loss"] < best_loss
            ):
                best = record["validation_exact"]
                best_loss = record["teacher_loss"]
                model.save(
                    output / "best.json", contract=contract, training_state=state
                )
        history.append(record)
        print(record, flush=True)
    return {
        "history": history,
        "training_seconds": time.perf_counter() - started,
        "decisions_per_epoch": len(targets),
        "best_validation_exact": best,
    }


def run(
    output, *, epochs=40, seed=0, evaluate_test=False, toy_report=None, polish_steps=160
):
    require_p2(toy_report or ROOT / "artifacts/toy-signed/report.json")
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    output = Path(output)
    splits = code_cases_v2()
    if (output / "hidden.json").exists():
        raise ValueError(
            "This study already opened hidden evaluation; do not retrain it"
        )
    manifest = lock_manifest(
        output / "dataset.json",
        splits=splits,
        graph=graph,
        ir_hash=spec_hash(),
        seed=314159,
        kind=DATA_VERSION,
    )
    contract = {
        "dataset_sha256": manifest["sha256"],
        "ir_version": IR_VERSION,
        "ir_sha256": spec_hash(),
    }
    model = FlyCodeV2(graph, seed=seed)
    initial = model.brain.weights_hash
    training = fit(model, splits, output, contract, epochs=epochs)
    best, state = NeuralSequence.restore(
        output / "best.json", graph, CODE_VOCABULARY, role="fly-code", contract=contract
    )
    validation = evaluate_code(best, splits["validation"])
    report = {
        "phase": "P3-v2",
        "provenance": provenance(ROOT),
        "seed": seed,
        "contract": contract,
        "initial_weights_sha256": initial,
        "final_weights_sha256": best.brain.weights_hash,
        "changed_edges": int(
            np.count_nonzero(best.brain.weights != best.brain.initial_weights)
        ),
        "training": training,
        "selected_epoch": state["epoch"],
        "validation": validation,
        "test_evaluated": False,
        "gate": {
            "passed": False,
            "reason": "hidden evaluation not yet authorized by validation",
        },
    }
    write_report(output / "report.json", report)
    if polish_steps:
        report = polish_saved(output, steps=polish_steps)
    if evaluate_test:
        report = open_hidden(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/code-v2")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--polish-steps", type=int, default=160)
    parser.add_argument(
        "--toy-report", type=Path, default=ROOT / "artifacts/toy-signed/report.json"
    )
    args = parser.parse_args()
    run(
        args.output,
        epochs=args.epochs,
        seed=args.seed,
        evaluate_test=args.evaluate_test,
        toy_report=args.toy_report,
        polish_steps=args.polish_steps,
    )


if __name__ == "__main__":
    main()
