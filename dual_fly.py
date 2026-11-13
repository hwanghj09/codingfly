"""Fly-only IR/Python and natural-language research CLI; legacy lab.py stays separate."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from evaluation.oracles import grade_code
from flybrain.graph import Graph, digest
from flybrain.policy import verify_generation_tree
from ir.grammar import validate
from ir.tokens import IR_VERSION, spec_hash
from models.fly_code import FlyCode
from models.pipeline import FlyPipeline, load_model
from models.retry import run_retries


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fly-only"], default="fly-only")
    parser.add_argument(
        "--engine",
        choices=["v1", "v2"],
        help="Default v2; explicit old checkpoint paths select v1",
    )
    parser.add_argument("--code-dir", type=Path, default=ROOT / "artifacts/code-v2")
    parser.add_argument("--nl-dir", type=Path, default=ROOT / "artifacts/nl-v3")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--graph", type=Path, default=ROOT / "flywire_lab_2048.csv")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ir", help="IR v0.1 text")
    source.add_argument(
        "--goal",
        help="Short supported Korean request; quote identifiers as JSON strings",
    )
    parser.add_argument(
        "--tests",
        type=Path,
        help="Optional JSON observations/function-call oracle for OS-isolated grading",
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="1..3 attempts; retries require --goal and --tests",
    )
    parser.add_argument(
        "--feedback-mode", choices=["weak", "supervised", "resample"], default="weak"
    )
    parser.add_argument("--retry-seed", type=int, default=0)
    args = parser.parse_args()
    engine = args.engine or ("v1" if args.checkpoint or args.dataset else "v2")
    try:
        if not 1 <= args.max_attempts <= 3:
            raise ValueError("--max-attempts must be between 1 and 3")
        if args.max_attempts > 1 and (args.goal is None or args.tests is None):
            raise ValueError("Retries require --goal and an explicit --tests oracle")
        verify_generation_tree()
        graph = Graph.from_csv(args.graph)
        expected = json.loads(args.tests.read_text()) if args.tests else None
        if args.goal is not None:
            if engine != "v2":
                raise ValueError(
                    "Natural-language generation requires the independent v2 Fly-NL model"
                )
            a = load_model(args.nl_dir, graph, role="fly-nl")
            b = load_model(
                args.code_dir,
                graph,
                role="fly-code",
                checkpoint=args.checkpoint,
                dataset=args.dataset,
            )
            result = run_retries(
                FlyPipeline(a, b),
                args.goal,
                expected=expected,
                record=args.trace,
                max_actions=args.max_actions,
                max_attempts=args.max_attempts,
                feedback_mode=args.feedback_mode,
                seed=args.retry_seed,
            )
            print(
                json.dumps(
                    {
                        "mode": "fly-only",
                        "stage": "P6-retry"
                        if args.max_attempts > 1
                        else "P4/P5-experiment",
                        **result,
                    },
                    ensure_ascii=False,
                )
            )
            return
        validate(args.ir)
        if engine == "v1":
            dataset = args.dataset or ROOT / "artifacts/fly-code/dataset.json"
            checkpoint = args.checkpoint or ROOT / "artifacts/fly-code/learned.json"
            manifest = json.loads(dataset.read_text())
            if manifest["sha256"] != digest(manifest["payload"]):
                raise ValueError("Dataset checksum mismatch")
            contract = {
                "dataset_sha256": manifest["sha256"],
                "ir_version": IR_VERSION,
                "ir_sha256": spec_hash(),
            }
            model, _ = FlyCode.restore(checkpoint, graph, contract=contract)
        else:
            model = load_model(
                args.code_dir,
                graph,
                role="fly-code",
                checkpoint=args.checkpoint,
                dataset=args.dataset,
            )
        generation = model.generate(
            args.ir, max_actions=args.max_actions, record=args.trace
        )
        result = {
            "mode": "fly-only",
            "stage": "P3-v2" if engine == "v2" else "P3-v1-experiment",
            **asdict(generation),
        }
        if expected is not None:
            result["verification"] = grade_code(generation.code, expected)
            result["passed"] = (
                generation.status == "eos" and result["verification"]["semantic_ok"]
            )
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"fly-only failed: {error}\n")


if __name__ == "__main__":
    main()
