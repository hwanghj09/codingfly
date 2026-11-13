"""Train expanded NL supervision in a new study; never reopen or overwrite v2."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.datasets import lock_manifest
from evaluation.nl_data_v3 import data_version, nl_cases_v3
from evaluation.report import provenance, write_report
from flybrain.graph import Graph
from ir.tokens import IR_VERSION, spec_hash
from models.pipeline import load_model
from models.train_nl import evaluate_nl, nl_teacher, nl_validation, require_p3
from models.train_sequence import encode_training, fit, polish


ROOT = Path(__file__).resolve().parents[1]


def require_unopened(output):
    output = Path(output)
    if (output / "hidden.json").exists() or (
        output / "evaluation-started.json"
    ).exists():
        raise ValueError(
            "This v3 study opened its test; no further tuning or evaluation"
        )
    if (output / "report.json").exists():
        if json.loads((output / "report.json").read_text()).get("test_evaluated"):
            raise ValueError(
                "This v3 study opened its test; no further tuning or evaluation"
            )


def validation_key(result):
    return (result["overall_semantic_exact_rate"], result["unknown_correct_rate"])


def polish_study(output, *, steps=160):
    output = Path(output)
    require_unopened(output)
    report = json.loads((output / "report.json").read_text())
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    model = load_model(output, graph, role="fly-nl")
    profile = report.get("profile", "initial")
    splits = nl_cases_v3(profile=profile)
    manifest = json.loads((output / "dataset.json").read_text())
    lock_manifest(
        output / "dataset.json",
        splits=splits,
        graph=graph,
        ir_hash=spec_hash(),
        seed=314159,
        kind=data_version(profile),
    )
    if manifest["sha256"] != report["contract"]["dataset_sha256"]:
        raise ValueError("Training report/dataset mismatch")
    stimuli, targets = encode_training(model, splits["train"], output, nl_teacher)
    result = polish(model, stimuli, targets, steps=steps)
    validation = evaluate_nl(model, splits["validation"])
    accepted = validation_key(validation) >= validation_key(report["validation"])
    result["accepted_by_validation"] = accepted
    if accepted:
        model.save(
            output / "best.json",
            contract=report["contract"],
            training_state={"stage": "v3-polish", "polishing": result},
        )
        report.update(
            validation=validation,
            selected_stage="v3-polish",
            final_weights_sha256=model.brain.weights_hash,
        )
    parent = load_model(report["parent_directory"], graph, role="fly-nl")
    if parent.brain.weights_hash != report["parent_weights_sha256"]:
        raise ValueError("Parent checkpoint changed during this study")
    selected = model if accepted else load_model(output, graph, role="fly-nl")
    report["changed_edges_from_parent"] = int(
        np.count_nonzero(selected.brain.weights != parent.brain.weights)
    )
    report.setdefault("polishing", []).append(result)
    report["provenance"] = provenance(ROOT)
    write_report(output / "report.json", report)
    print(
        "v3 polished validation",
        validation_key(report["validation"]),
        result,
        flush=True,
    )
    return report


def train(
    output,
    parent,
    p3_report,
    *,
    epochs=20,
    polish_steps=160,
    profile="morphology",
    context_bag=True,
):
    output, parent = Path(output), Path(parent)
    require_unopened(output)
    if output.resolve() == parent.resolve():
        raise ValueError("v3 must use a new study directory")
    if (output / "started.json").exists() or (output / "report.json").exists():
        raise ValueError("Study already exists; use polish for a still-unopened study")
    p3 = require_p3(p3_report)
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    model = load_model(parent, graph, role="fly-nl")
    parent_hash = model.brain.weights_hash
    parent_weights = model.brain.weights.copy()
    splits = nl_cases_v3(profile=profile)
    manifest = lock_manifest(
        output / "dataset.json",
        splits=splits,
        graph=graph,
        ir_hash=spec_hash(),
        seed=314159,
        kind=data_version(profile),
    )
    contract = {
        "dataset_sha256": manifest["sha256"],
        "ir_version": IR_VERSION,
        "ir_sha256": spec_hash(),
    }
    write_report(
        output / "started.json",
        {
            "dataset_sha256": manifest["sha256"],
            "parent_weights_sha256": parent_hash,
            "epochs": epochs,
            "polish_steps": polish_steps,
            "provenance": provenance(ROOT),
            "selection": "validation only; test requires a separate explicit command",
        },
    )
    parent_validation = evaluate_nl(model, splits["validation"])
    if context_bag:
        model.encoder = model.make_encoder(
            graph,
            model.brain.input_nodes,
            model.brain.seed,
            model.role,
            context_bag=True,
        )
    baseline = (
        evaluate_nl(model, splits["validation"]) if context_bag else parent_validation
    )
    print("parent validation", validation_key(baseline), flush=True)
    model.save(
        output / "warm-start.json",
        contract=contract,
        training_state={"parent_weights_sha256": parent_hash},
    )
    training = fit(
        model,
        splits,
        output,
        contract,
        epochs=epochs,
        learning_rate=0.002,
        batch_size=128,
        action_builder=nl_teacher,
        validation_scorer=nl_validation,
    )
    selected = load_model(output, graph, role="fly-nl")
    validation = evaluate_nl(selected, splits["validation"])
    stage = "v3-Adam"
    if validation_key(validation) < validation_key(baseline):
        selected = load_model(
            output, graph, role="fly-nl", checkpoint=output / "warm-start.json"
        )
        selected.save(
            output / "best.json", contract=contract, training_state={"stage": "parent"}
        )
        validation, stage = baseline, "parent"
    report = {
        "phase": "Fly-NL-v3-accuracy",
        "profile": profile,
        "context_bag": context_bag,
        "contract": contract,
        "p3_gate": p3,
        "parent_directory": str(parent),
        "parent_weights_sha256": parent_hash,
        "seed": selected.brain.seed,
        "training": training,
        "baseline_validation": parent_validation,
        "warm_start_validation": baseline,
        "validation": validation,
        "selected_stage": stage,
        "final_weights_sha256": selected.brain.weights_hash,
        "test_evaluated": False,
        "provenance": provenance(ROOT),
        "changed_edges_from_parent": int(
            np.count_nonzero(selected.brain.weights != parent_weights)
        ),
        "limits": [
            "Eight supported tasks; new combinations of known Korean words",
            "v2 test phrases are public training material in v3; historical scores are regression only",
            "One warm-start seed; no claim of independent training-seed replication",
        ],
    }
    write_report(output / "report.json", report)
    if polish_steps:
        report = polish_study(output, steps=polish_steps)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/nl-v3")
    parser.add_argument("--parent", type=Path, default=ROOT / "artifacts/nl-v2")
    parser.add_argument(
        "--p3-report", type=Path, default=ROOT / "artifacts/code-v2/report.json"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--polish-steps", type=int, default=160)
    parser.add_argument("--polish-only", action="store_true")
    parser.add_argument(
        "--profile",
        choices=["initial", "contrastive", "morphology"],
        default="morphology",
    )
    parser.add_argument("--legacy-context", action="store_true")
    args = parser.parse_args()
    if args.polish_only:
        polish_study(args.output, steps=args.polish_steps)
    else:
        train(
            args.output,
            args.parent,
            args.p3_report,
            epochs=args.epochs,
            polish_steps=args.polish_steps,
            profile=args.profile,
            context_bag=not args.legacy_context,
        )


if __name__ == "__main__":
    main()
