"""Supervised backpropagation through time, restricted to existing directed edges.

This is an engineering learning rule, not evidence for biological backpropagation.
The signed descent eligibility is accumulated transiently and checkpointed for audit.
"""

import math

import numpy as np


def supervised_step(brain, stimulus, expected, *, learning_rate=0.03, frozen=False):
    if (
        type(expected) is not int
        or not 0 <= expected < len(brain.output_nodes)
        or not math.isfinite(learning_rate)
        or not 0 < learning_rate <= 1
    ):
        raise ValueError("Invalid learning target or rate")
    probabilities, tape, _ = brain.forward(stimulus)
    loss = -math.log(max(float(probabilities[expected]), 1e-300))
    error = probabilities.copy()
    error[expected] -= 1.0
    adjoint = np.zeros_like(brain.state)
    adjoint[list(brain.output_nodes)] = error * brain.config.output_scale
    gradient = np.zeros_like(brain.weights)
    cfg = brain.config
    for previous, proposal in reversed(tape):
        local = adjoint * cfg.leak * (1.0 - proposal * proposal)
        gradient += cfg.gain * previous[brain.pre] * local[brain.post]
        adjoint = (1.0 - cfg.leak) * adjoint + np.bincount(
            brain.pre,
            weights=cfg.gain * brain.weights * local[brain.post],
            minlength=len(brain.state),
        )
    old = brain.weights.copy()
    if not frozen:
        brain.eligibility = -gradient
        # The configured artificial sign law always bounds original edge strengths.
        candidate = brain.weights - learning_rate * gradient
        brain.weights = (
            brain.signs * np.clip(candidate * brain.signs, 0.0, cfg.max_weight)
            if cfg.sign_constraint
            else np.clip(candidate, -cfg.max_weight, cfg.max_weight)
        )
        brain.updates += 1
    brain.steps_seen += cfg.steps
    delta = brain.weights - old
    return {
        "loss": loss,
        "chosen": int(np.argmax(probabilities)),
        "changed_edges": int(np.count_nonzero(delta)),
        "max_abs_delta": float(np.max(np.abs(delta), initial=0.0)),
        "mean_abs_delta": float(np.mean(np.abs(delta))) if len(delta) else 0.0,
        "clipped_edges": int(np.count_nonzero(np.abs(brain.weights) == cfg.max_weight)),
    }
