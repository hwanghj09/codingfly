"""Copy evaluated native checkpoints into the versioned browser release."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flybrain.graph import Graph
from ir.tokens import IR_VERSION, spec_hash
from models.pipeline import load_model


def export(nl_dir, code_dir, report_path, destination):
    destination = Path(destination)
    report = json.loads(Path(report_path).read_text())
    if not report.get("test_evaluated") or not report.get("restore_exact"):
        raise ValueError("An evaluated and restore-verified model is required")
    graph = Graph.from_csv(ROOT / "flywire_lab_2048.csv")
    release = {
        "version": 1,
        "name": "Fly-NL v3.2 + Fly-Code v2",
        "graph_sha256": graph.topology_hash,
        "graph_source_sha256": graph.source_hash,
        "ir_version": IR_VERSION,
        "ir_sha256": spec_hash(),
        "supported_tasks": 8,
        "evaluation": {
            "correct": report["hidden"]["correct"],
            "total": report["hidden"]["supported_total"],
            "unsupported_correct": report["hidden"]["unsupported_correct"],
            "unsupported_total": report["hidden"]["unsupported_total"],
        },
        "models": {},
    }
    destination.mkdir(parents=True, exist_ok=True)
    for role, directory, key in (
        ("fly-nl", nl_dir, "final_weights_sha256"),
        ("fly-code", code_dir, "b_weights_sha256"),
    ):
        model = load_model(directory, graph, role=role)
        if model.brain.weights_hash != report[key]:
            raise ValueError("Checkpoint weights differ from evaluated weights")
        source = Path(directory) / "best.json"
        target = destination / f"{role}.json"
        shutil.copyfile(source, target)
        payload = json.loads(target.read_text())["payload"]
        release["models"][role] = {
            "file": f"checkpoints/{role}.json",
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "weights_sha256": model.brain.weights_hash,
            "contract": {
                k: payload["contract"][k]
                for k in ("dataset_sha256", "ir_version", "ir_sha256")
            },
            "encoder": payload["encoder"],
        }
    (destination / "release.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nl-dir", type=Path, default=ROOT / "artifacts/nl-v3")
    parser.add_argument("--code-dir", type=Path, default=ROOT / "artifacts/code-v2")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "docs/reports/accuracy-v3.json"
    )
    parser.add_argument("--destination", type=Path, default=ROOT / "web/checkpoints")
    args = parser.parse_args()
    export(args.nl_dir, args.code_dir, args.report, args.destination)
