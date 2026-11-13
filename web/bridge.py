"""Browser adapter to the unchanged neural generation pipeline. No teachers or examples."""

import ast
import json
from pathlib import Path

from flybrain.graph import Graph
from ir.tokens import IR_VERSION, spec_hash
from models.fly_code_v2 import CODE_VOCABULARY
from models.fly_nl import NL_VOCABULARY
from models.neural_sequence import NeuralSequence
from models.pipeline import FlyPipeline


def initialize():
    release = json.loads(Path("browser_manifest.json").read_text())
    graph = Graph.from_csv("flywire_lab_2048.csv")
    if (
        graph.topology_hash != release["graph_sha256"]
        or spec_hash() != release["ir_sha256"]
        or IR_VERSION != release["ir_version"]
    ):
        raise ValueError("Browser model contract mismatch")
    models = []
    for role, vocabulary in (("fly-nl", NL_VOCABULARY), ("fly-code", CODE_VOCABULARY)):
        entry = release["models"][role]
        model, _ = NeuralSequence.restore(
            entry["file"], graph, vocabulary, role=role, contract=entry["contract"]
        )
        if model.brain.weights_hash != entry["weights_sha256"]:
            raise ValueError("Unexpected browser synapses")
        models.append(model)
    return FlyPipeline(*models)


PIPELINE = initialize()


def generate(goal):
    if not isinstance(goal, str) or not goal.strip() or len(goal.encode()) > 2048:
        raise ValueError("요청을 2,048바이트 이내로 입력해 주세요.")
    result = PIPELINE.run(goal, max_actions=512)
    result["functions"] = []
    if result["status"] == "syntax_only":
        result["functions"] = [
            {"name": node.name, "parameters": [a.arg for a in node.args.args]}
            for node in ast.parse(result["code"]).body
            if isinstance(node, ast.FunctionDef)
        ]
    return json.dumps(result, ensure_ascii=False, allow_nan=False)
