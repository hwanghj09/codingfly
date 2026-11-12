"""Fly-Code v2: fixed lexical COPY candidates and multiplexed neural outputs."""

from flybrain.sequence_io import PYTHON_LITERALS, Vocabulary
from models.neural_sequence import NeuralSequence


CODE_VOCABULARY = Vocabulary(PYTHON_LITERALS)


class FlyCodeV2(NeuralSequence):
    def __init__(self, graph, *, seed=0, config=None):
        super().__init__(
            graph, CODE_VOCABULARY, role="fly-code", seed=seed, config=config
        )
