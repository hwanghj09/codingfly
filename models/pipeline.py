"""Fly-NL -> structural IR validation -> Fly-Code. Validators never rewrite output."""

import ast
from dataclasses import asdict
import json
from pathlib import Path
import uuid
import numpy as np

from evaluation.oracles import grade_code
from flybrain.graph import digest
from ir.grammar import InvalidIR, validate
from ir.tokens import IR_VERSION, spec_hash
from models.fly_code_v2 import CODE_VOCABULARY
from models.fly_nl import NL_VOCABULARY
from models.neural_sequence import NeuralSequence


def load_model(directory, graph, *, role, checkpoint=None, dataset=None):
    directory = Path(directory)
    manifest = json.loads(Path(dataset or directory / "dataset.json").read_text())
    if manifest["sha256"] != digest(manifest["payload"]):
        raise ValueError("Dataset checksum mismatch")
    contract = {
        "dataset_sha256": manifest["sha256"],
        "ir_version": IR_VERSION,
        "ir_sha256": spec_hash(),
    }
    vocabulary = CODE_VOCABULARY if role == "fly-code" else NL_VOCABULARY
    return NeuralSequence.restore(
        checkpoint or directory / "best.json",
        graph,
        vocabulary,
        role=role,
        contract=contract,
    )[0]


class FlyPipeline:
    def __init__(self, fly_nl, fly_code):
        if (
            fly_nl.role != "fly-nl"
            or fly_code.role != "fly-code"
            or fly_nl.brain is fly_code.brain
            or np.shares_memory(fly_nl.brain.weights, fly_code.brain.weights)
        ):
            raise ValueError("Pipeline requires independent A/B roles and synapses")
        self.fly_nl, self.fly_code = fly_nl, fly_code

    def run(
        self,
        goal,
        *,
        expected=None,
        record=False,
        max_actions=512,
        nl_options=None,
        code_options=None,
    ):
        first = self.fly_nl.generate(
            goal, record=record, max_actions=max_actions, **(nl_options or {})
        )
        return self.complete(
            first,
            expected=expected,
            record=record,
            max_actions=max_actions,
            code_options=code_options,
        )

    def complete(
        self, first, *, expected=None, record=False, max_actions=512, code_options=None
    ):
        """Validate an actual A generation, optionally reusing it for B-only retries."""
        result = {
            "trace_id": str(uuid.uuid4()),
            "status": "invalid_ir",
            "ir": first.code,
            "fly_nl": asdict(first),
            "code": "",
            "fly_code": None,
            "nl_weights_sha256": self.fly_nl.brain.weights_hash,
            "code_weights_sha256": self.fly_code.brain.weights_hash,
        }
        if first.status != "eos":
            result["status"] = "nl_generation_error"
            return result
        if first.code.strip() == "UNKNOWN":
            result["status"] = "uncertain"
            return result
        try:
            validate(first.code)
        except InvalidIR as error:
            result["error"] = str(error)
            return result
        second = self.fly_code.generate(
            first.code, record=record, max_actions=max_actions, **(code_options or {})
        )
        result.update(code=second.code, fly_code=asdict(second))
        if second.status != "eos":
            result["status"] = "code_generation_error"
            return result
        try:
            if not second.code.strip():
                raise SyntaxError("Empty program")
            ast.parse(second.code)
            compile(second.code, "<generated>", "exec")
        except (SyntaxError, ValueError, RecursionError) as error:
            result.update(status="syntax_error", error=str(error))
            return result
        result["status"] = "syntax_only"
        if expected is not None:
            result["verification"] = grade_code(second.code, expected)
            result["status"] = (
                "ok" if result["verification"]["semantic_ok"] else "test_failure"
            )
        return result
