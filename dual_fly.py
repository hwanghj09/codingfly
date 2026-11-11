"""Fly-only research CLI. Existing `lab.py`/`fly.py` remain the explicit legacy demo."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from flybrain.graph import Graph, digest
from flybrain.policy import verify_generation_tree
from ir.grammar import validate
from ir.tokens import IR_VERSION, spec_hash
from models.fly_code import FlyCode


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fly-only"], default="fly-only")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/fly-code/learned.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/fly-code/dataset.json")
    parser.add_argument("--graph", type=Path, default=ROOT / "flywire_lab_2048.csv")
    parser.add_argument("--ir", required=True, help="IR v0.1 text; Fly-NL is not yet validated")
    parser.add_argument("--trace", action="store_true", help="Include neuron/weight evidence for every action")
    parser.add_argument("--max-actions", type=int, default=64)
    args = parser.parse_args()
    try:
        verify_generation_tree()
        validate(args.ir)
        manifest = json.loads(args.dataset.read_text(encoding="utf-8"))
        if manifest["sha256"] != digest(manifest["payload"]):
            raise ValueError("Dataset checksum mismatch")
        contract = {"dataset_sha256": manifest["sha256"], "ir_version": IR_VERSION, "ir_sha256": spec_hash()}
        model, _ = FlyCode.restore(args.checkpoint, Graph.from_csv(args.graph), contract=contract)
        generation = model.generate(args.ir, max_actions=args.max_actions, record=args.trace)
        print(json.dumps({"mode": "fly-only", "stage": "P3-experiment", **asdict(generation)}, ensure_ascii=False))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"fly-only failed: {error}\n")


if __name__ == "__main__":
    main()
