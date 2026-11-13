"""Independent neural Fly-NL. No dictionary, regex intent parsing or external model."""

from flybrain.sequence_io import IR_LITERALS, Vocabulary
from models.neural_sequence import NeuralSequence


NL_VOCABULARY = Vocabulary(
    IR_LITERALS, inner_copy=True, context_slots=8, register_slots=8
)


class FlyNL(NeuralSequence):
    def __init__(self, graph, *, seed=101, config=None, context_bag=False):
        super().__init__(
            graph,
            NL_VOCABULARY,
            role="fly-nl",
            seed=seed,
            config=config,
            context_bag=context_bag,
        )
