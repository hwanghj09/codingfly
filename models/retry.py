"""At most three neural attempts with observable failure feedback; no online learning."""

from dataclasses import asdict

from flybrain.graph import digest
from flybrain.retry_input import RETRY_IO_VERSION, RetryDecoding
from ir.grammar import InvalidIR
from ir.serializer import canonicalize
from models.fly_code import Generation


STOP_STATUSES = {"ok", "syntax_only", "uncertain"}
INFRASTRUCTURE = {"sandbox_unavailable", "sandbox_error"}


def same_ir(actual, expected):
    try:
        return canonicalize(actual) == canonicalize(expected)
    except (InvalidIR, ValueError):
        return False


def failed_checks(verification, expected):
    """Only public-oracle failures; no expected answer values enter the model."""
    checks = []
    if verification.get("status") != "ok":
        return [
            {
                "kind": "execution",
                "status": verification.get("status"),
                "error": str(verification.get("error", ""))[:256],
            }
        ]
    if verification.get("stdout") != expected.get("stdout", ""):
        checks.append(
            {"kind": "stdout", "actual": str(verification.get("stdout"))[:256]}
        )
    actual_variables = verification.get("variables", {})
    for name, value in expected.get("variables", {}).items():
        if name not in actual_variables or actual_variables[name] != value:
            checks.append(
                {
                    "kind": "variable",
                    "name": name,
                    "actual": repr(actual_variables.get(name))[:128],
                }
            )
    returns = verification.get("returns", [])
    for index, value in enumerate(expected.get("returns", [])):
        if index >= len(returns) or returns[index] != value:
            call = (
                expected.get("calls", [])[index]
                if index < len(expected.get("calls", []))
                else {}
            )
            checks.append(
                {
                    "kind": "return",
                    "index": index,
                    "input": repr(call)[:256],
                    "actual": repr(returns[index] if index < len(returns) else None)[
                        :128
                    ],
                }
            )
    return checks[:32]


def failure_signal(result, expected, *, mode, expected_ir=None):
    """Supervision attributes responsibility; it never supplies replacement IR/code."""
    status = result["status"]
    if mode in {"weak", "resample"}:
        return {"reward": -1, "stage": "unlocalized", "issue": "validation_failed"}
    if expected_ir is not None and not same_ir(result["ir"], expected_ir):
        stage = "A"
    elif status in {"invalid_ir", "nl_generation_error"}:
        stage = "A"
    elif expected_ir is not None or status in {"syntax_error", "code_generation_error"}:
        stage = "B"
    else:
        stage = "unlocalized"
    return {
        "reward": -1,
        "stage": stage,
        "issue": status,
        "error": str(result.get("error", ""))[:256],
        "failed_checks": failed_checks(result["verification"], expected)
        if "verification" in result
        else [],
    }


def run_retries(
    pipeline,
    goal,
    *,
    expected=None,
    expected_ir=None,
    feedback_mode="weak",
    max_attempts=3,
    seed=0,
    record=False,
    max_actions=512,
):
    """First attempt stays greedy. Retries explore neural context ranks, B digit scores.

    `resample` is the equal-budget no-feedback control. `supervised` may receive gold
    IR for attribution only; ordinary CLI use supplies no gold and marks semantic
    responsibility unlocalized. No oracle is passed to a model's generate method.
    """
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("Retry budget must be between 1 and 3")
    if feedback_mode not in {"resample", "weak", "supervised"}:
        raise ValueError("Unknown feedback mode")
    if type(seed) is not int or not 0 <= seed < 2**32 - 10:
        raise ValueError("Invalid retry seed")
    if max_attempts > 1 and expected is None:
        raise ValueError("Retries require an explicit public test oracle")
    if expected_ir is not None and feedback_mode != "supervised":
        raise ValueError("Gold IR is only allowed for supervised attribution")
    before = (pipeline.fly_nl.brain.weights_hash, pipeline.fly_code.brain.weights_hash)
    attempts, signal, result = [], None, None
    a_retries = 0
    for index in range(max_attempts):
        feedback = None
        if index and feedback_mode != "resample":
            feedback = {
                **signal,
                "previous_ir_sha256": digest(result["ir"]),
                "previous_code_sha256": digest(result["code"]),
            }
        reuse_a = bool(index and signal["stage"] == "B")
        nl_options, code_options = {}, {}
        if index:
            # Only the blamed role is given its previous action history. An A
            # change necessarily requires fresh B translation, without blaming B.
            if not reuse_a:
                a_retries += 1
                nl_options = {
                    "decoding": RetryDecoding(
                        context_rank=a_retries, seed=seed + index
                    ),
                    "feedback": {
                        **feedback,
                        "previous_actions": result["fly_nl"]["actions"][-128:],
                    }
                    if feedback
                    else None,
                }
            if signal["stage"] != "A":
                code_options = {
                    "decoding": RetryDecoding(temperature=0.3, seed=seed + index),
                    "feedback": {
                        **feedback,
                        "previous_actions": (result.get("fly_code") or {}).get(
                            "actions", []
                        )[-128:],
                    }
                    if feedback
                    else None,
                }
        if reuse_a:
            previous = result["fly_nl"]
            first = Generation(
                previous["code"],
                previous["status"],
                tuple(previous["actions"]),
                tuple(previous["trace"]),
            )
            result = pipeline.complete(
                first,
                expected=expected,
                record=record,
                max_actions=max_actions,
                code_options=code_options,
            )
        else:
            result = pipeline.run(
                goal,
                expected=expected,
                record=record,
                max_actions=max_actions,
                nl_options=nl_options,
                code_options=code_options,
            )
        entry = {
            "number": index + 1,
            "reused_a": reuse_a,
            "nl_decoding": asdict(nl_options["decoding"]) if nl_options else None,
            "code_decoding": asdict(code_options["decoding"]) if code_options else None,
            "input_feedback": feedback,
            "result": result,
            "reward": 1
            if result["status"] == "ok"
            else 0
            if result["status"] in STOP_STATUSES
            else -1,
        }
        attempts.append(entry)
        if result["status"] in STOP_STATUSES:
            break
        if result.get("verification", {}).get("status") in INFRASTRUCTURE:
            break
        signal = failure_signal(
            result, expected or {}, mode=feedback_mode, expected_ir=expected_ir
        )
        entry["failure_feedback"] = signal
    after = (pipeline.fly_nl.brain.weights_hash, pipeline.fly_code.brain.weights_hash)
    if before != after:
        raise RuntimeError("Inference retries must never modify synaptic weights")
    return {
        **result,
        "retry": {
            "protocol": RETRY_IO_VERSION,
            "feedback_mode": feedback_mode,
            "supervisor_has_gold_ir": expected_ir is not None,
            "max_attempts": max_attempts,
            "attempts_used": len(attempts),
            "seed": seed,
            "weights_unchanged": True,
            "learning_updates": 0,
            "a_retry_count": a_retries,
            "b_retry_count": sum(
                a["result"]["fly_code"] is not None for a in attempts[1:]
            ),
            "attempts": attempts,
        },
    }
