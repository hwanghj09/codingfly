"""Sparse batched BPTT and Adam; all optimized arrays index original edges only."""

import numpy as np
from scipy.sparse import csr_matrix


def loss_gradient(brain, stimuli, targets):
    stimuli = np.asarray(stimuli, dtype=np.float64)
    targets = np.asarray(targets)
    n = len(brain.state)
    if (
        stimuli.ndim != 2
        or stimuli.shape[1] != n
        or len(stimuli) != len(targets)
        or not len(targets)
        or targets.dtype.kind not in "iu"
        or np.any(targets < 0)
        or np.any(targets >= len(brain.output_nodes))
        or not np.isfinite(stimuli).all()
        or np.any(stimuli[:, brain.output_nodes])
    ):
        raise ValueError("Invalid neural batch")
    matrix = csr_matrix((brain.weights, (brain.post, brain.pre)), shape=(n, n))
    state = np.zeros_like(stimuli)
    tape, cfg = [], brain.config
    for _ in range(cfg.steps):
        previous = state
        proposal = np.tanh(stimuli + cfg.gain * matrix.dot(previous.T).T)
        state = (1 - cfg.leak) * previous + cfg.leak * proposal
        tape.append((previous, proposal))
    logits = cfg.output_scale * state[:, brain.output_nodes]
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    loss = float(
        -np.log(
            np.maximum(probabilities[np.arange(len(targets)), targets], 1e-300)
        ).mean()
    )
    errors = probabilities.copy()
    errors[np.arange(len(targets)), targets] -= 1
    adjoint, gradient = np.zeros_like(state), np.zeros_like(brain.weights)
    adjoint[:, brain.output_nodes] = cfg.output_scale * errors / len(targets)
    for previous, proposal in reversed(tape):
        local = cfg.leak * adjoint * (1 - proposal**2)
        gradient += cfg.gain * np.sum(
            previous[:, brain.pre] * local[:, brain.post], axis=0
        )
        adjoint = (1 - cfg.leak) * adjoint + cfg.gain * matrix.T.dot(local.T).T
    return loss, gradient, probabilities


class EdgeAdam:
    def __init__(self, brain, *, learning_rate=0.02):
        if not 0 < learning_rate <= 1:
            raise ValueError("Invalid learning rate")
        self.brain, self.learning_rate = brain, learning_rate
        self.first, self.second = (
            np.zeros_like(brain.weights),
            np.zeros_like(brain.weights),
        )
        self.step = 0

    def update(self, stimuli, targets, *, frozen=False):
        loss, gradient, probabilities = loss_gradient(self.brain, stimuli, targets)
        brain = self.brain
        if not frozen:
            self.step += 1
            self.first = 0.9 * self.first + 0.1 * gradient
            self.second = 0.999 * self.second + 0.001 * gradient**2
            direction = (
                self.first
                / (1 - 0.9**self.step)
                / (np.sqrt(self.second / (1 - 0.999**self.step)) + 1e-8)
            )
            candidate = brain.weights - self.learning_rate * direction
            brain.weights = (
                brain.signs
                * np.clip(candidate * brain.signs, 0.0, brain.config.max_weight)
                if brain.config.sign_constraint
                else np.clip(
                    candidate, -brain.config.max_weight, brain.config.max_weight
                )
            )
            brain.eligibility = -gradient
            brain.updates += 1
        brain.steps_seen += brain.config.steps * len(targets)
        return {
            "loss": loss,
            "correct": int(np.sum(probabilities.argmax(axis=1) == targets)),
        }

    def state(self):
        return {
            "first": self.first.tolist(),
            "second": self.second.tolist(),
            "step": self.step,
            "learning_rate": self.learning_rate,
        }

    def restore(self, state):
        first, second = np.array(state["first"]), np.array(state["second"])
        if (
            first.shape != self.brain.weights.shape
            or second.shape != first.shape
            or not np.isfinite(first).all()
            or not np.isfinite(second).all()
            or np.any(second < 0)
            or type(state["step"]) is not int
            or state["step"] < 0
            or state["learning_rate"] != self.learning_rate
        ):
            raise ValueError("Invalid edge optimizer state")
        self.first, self.second, self.step = first, second, state["step"]
