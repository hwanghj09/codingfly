"""Fixed retry I/O: no language interpretation, learned coefficients or answer repair."""

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


RETRY_IO_VERSION = "failure-bytes-and-neural-ranking-v1"


@dataclass(frozen=True)
class RetryDecoding:
    """Bounded search over neural scores, independent of the execution oracle."""

    context_rank: int = 0
    temperature: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if (
            type(self.context_rank) is not int
            or not 0 <= self.context_rank < 8
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("Invalid retry decoding bounds")

    def choose(self, probabilities, rng):
        if not self.temperature:
            return int(np.argmax(probabilities))
        logits = np.log(np.maximum(probabilities, 1e-15)) / self.temperature
        weights = np.exp(logits - logits.max())
        return int(rng.choice(len(weights), p=weights / weights.sum()))


def feedback_drive(encoder, feedback):
    """Hash bounded failure bytes onto input neurons; never touch readout neurons."""
    if feedback is None:
        return None, None
    raw = json.dumps(
        feedback,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if len(raw) > 16384:
        raise ValueError("Oversized retry feedback")
    # Hash fixed-size chunks; no words, error meanings or task labels are interpreted.
    features = np.array(
        [
            int.from_bytes(
                hashlib.blake2b(raw[i : i + 32], digest_size=8).digest(), "little"
            )
            for i in range(0, len(raw), 32)
        ],
        dtype=np.uint64,
    )
    drive = 0.01 * encoder.project(features, 0x7A368D5B)
    return drive.astype(np.float32), hashlib.sha256(raw).hexdigest()
