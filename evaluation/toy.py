"""P2 learning gate. python -m evaluation.toy --output artifacts/toy"""

import argparse
from dataclasses import asdict
from pathlib import Path
import random
import time

import numpy as np

from evaluation.ablation import random_graph, rewire
from evaluation.datasets import lock_manifest, toy_cases
from evaluation.report import provenance, summary, write_report
from flybrain.checkpoint import load, save
from flybrain.dynamics import Brain, DynamicsConfig
from flybrain.encoder import ByteEncoder
from flybrain.graph import Graph, digest
from flybrain.plasticity import supervised_step


ROOT = Path(__file__).resolve().parents[1]
TOY_VERSION = "first-byte-with-unseen-noise-v1"


def accuracy(brain, encoded, *, transmission=True):
    return sum(int(np.argmax(brain.forward(stimulus, transmission=transmission)[0])) == label
               for stimulus, label in encoded) / len(encoded)


def run_condition(graph, splits, *, seed, condition, epochs, learning_rate, config, output, contract):
    brain = Brain(graph, 4, seed=seed, config=config)
    encoder = ByteEncoder(len(graph.node_ids), brain.input_nodes, seed=seed)
    encoded = {name: [(encoder.encode(bytes.fromhex(case["source"])), int(case["target"]))
                      for case in cases] for name, cases in splits.items()}
    initial = brain.weights_hash
    before = accuracy(brain, encoded["validation"])
    training = []
    order = list(range(len(encoded["train"])))
    rng = random.Random(seed)
    started = time.perf_counter()
    for epoch in range(epochs):
        rng.shuffle(order)
        losses, deltas = [], []
        for i in order:
            stimulus, expected = encoded["train"][i]
            result = supervised_step(brain, stimulus, expected, learning_rate=learning_rate,
                                     frozen=condition == "frozen")
            losses.append(result["loss"])
            deltas.append(result["mean_abs_delta"])
        training.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses)),
                         "mean_abs_delta": float(np.mean(deltas)),
                         "validation_accuracy": accuracy(brain, encoded["validation"])})
    # Fixed budget: neither test labels nor validation choose epochs or checkpoints.
    elapsed = time.perf_counter() - started
    final = accuracy(brain, encoded["test"])
    checkpoint = output / f"{condition}-{seed}.json"
    save(brain, checkpoint, role="toy", contract=contract,
         training_state={"epochs": epochs, "shuffle_rng_state": rng.getstate()})
    restored, _ = load(checkpoint, graph, role="toy", contract=contract)
    restore_exact = all(np.array_equal(brain.forward(stimulus)[0], restored.forward(stimulus)[0])
                        for stimulus, _ in encoded["test"])
    changed = np.abs(brain.weights - brain.initial_weights)
    _, _, trace = brain.forward(encoded["test"][0][0], record=True)
    return {"seed": seed, "condition": condition, "config": asdict(config),
            "graph_sha256": graph.topology_hash,
            "connectivity": graph.connectivity(brain.input_nodes, brain.output_nodes),
            "initial_weights_sha256": initial,
            "final_weights_sha256": brain.weights_hash,
            "before_validation_accuracy": before, "test_accuracy": final,
            "cut_test_accuracy": accuracy(brain, encoded["test"], transmission=False),
            "changed_edges": int(np.count_nonzero(changed)),
            "weight_delta_quantiles": np.quantile(changed, [0, .25, .5, .75, 1]).tolist(),
            "restore_exact": restore_exact, "updates": brain.updates,
            "training_seconds": elapsed, "history": training, "example_trace": trace}


def run(output, *, graph_path=ROOT / "flywire_sample.csv", seeds=range(5), epochs=30,
        learning_rate=.03, controls=True):
    if epochs < 1 or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Positive epochs and distinct seeds required")
    graph = Graph.from_csv(graph_path)
    output = Path(output)
    splits = toy_cases()
    manifest = lock_manifest(output / "dataset.json", splits=splits, graph=graph,
                             ir_hash=digest(TOY_VERSION), seed=1729, kind=TOY_VERSION)
    contract = {"dataset_sha256": manifest["sha256"], "task": TOY_VERSION}
    config = DynamicsConfig()
    runs = []
    for seed in seeds:
        conditions = [("learned", graph, config), ("frozen", graph, config)]
        if controls:
            conditions.extend([("rewired", rewire(graph, seed), config),
                               ("random", random_graph(graph, seed), config),
                               ("inhibitory20", graph, DynamicsConfig(inhibitory_fraction=.2))])
        for condition, candidate, cfg in conditions:
            result = run_condition(candidate, splits, seed=seed, condition=condition,
                                   epochs=epochs, learning_rate=learning_rate, config=cfg,
                                   output=output, contract=contract)
            runs.append(result)
            print(f"seed={seed} {condition:12} test={result['test_accuracy']:.3f} "
                  f"changed={result['changed_edges']} restore={result['restore_exact']}", flush=True)
    pairs = [(next(r for r in runs if r["seed"] == seed and r["condition"] == "learned"),
              next(r for r in runs if r["seed"] == seed and r["condition"] == "frozen")) for seed in seeds]
    passed = len(seeds) >= 5 and all(
        learned["test_accuracy"] >= .75 and learned["test_accuracy"] > frozen["test_accuracy"]
        and learned["changed_edges"] > 0 and frozen["changed_edges"] == 0
        and learned["restore_exact"] and frozen["restore_exact"]
        and learned["cut_test_accuracy"] <= .25
        and learned["initial_weights_sha256"] == frozen["initial_weights_sha256"]
        for learned, frozen in pairs)
    report = {"version": 1, "phase": "P2", "provenance": provenance(ROOT),
              "graph": graph.metadata(), "dataset_sha256": manifest["sha256"],
              "learning_rule": "supervised BPTT on existing edges; sign-preserving projection",
              "learning_rate": learning_rate, "epochs": epochs,
              "gate": {"passed": passed, "minimum_seeds": 5, "min_each_learned_accuracy": .75,
                       "rule": "all seeds: learned > frozen, learned >= .75; changed learned / unchanged frozen; exact restore; cut <= chance",
                       "next": "P3 experiment permitted" if passed else "STOP before P3; investigate neural engine"},
              "summary": {name: summary([r["test_accuracy"] for r in runs if r["condition"] == name])
                          for name in sorted({r["condition"] for r in runs})},
              "runs": runs, "limits": ["Toy byte classification only; no language/code generation claim",
                                         "Noise-group holdout is not an algorithm-family holdout",
                                         "Rewired/random controls use their own fixed indegree readouts"]}
    write_report(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/toy")
    parser.add_argument("--graph", type=Path, default=ROOT / "flywire_sample.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--learning-rate", type=float, default=.03)
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args()
    report = run(args.output, graph_path=args.graph, seeds=args.seeds, epochs=args.epochs,
                 learning_rate=args.learning_rate, controls=not args.no_controls)
    print(report["gate"])


if __name__ == "__main__":
    main()
