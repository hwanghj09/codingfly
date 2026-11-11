import math
import random
import re
from collections import Counter

from algorithm_phrases import PHRASES
from reward import RewardMemory


def normalize_vector(vector):
    norm = math.hypot(*vector.values())
    return {key: value / norm for key, value in vector.items() if value}


def similarity(left, right):
    return sum(value * right.get(key, 0) for key, value in left.items())


class AlgorithmModel:
    def __init__(self, fly, *, lexical=True):
        self.fly = fly
        self.lexical = lexical
        self.examples = {}
        self.centroids = {}
        self.feature_classes = Counter(
            feature for phrases in PHRASES.values()
            for feature in {value for phrase in phrases for value in self.features(phrase)}
        )
        memory = RewardMemory(fly, [], [], {})
        rng = random.Random(42)
        actions = []
        for action, phrases in PHRASES.items():
            actions.append(action)
            vectors = []
            for phrase in phrases:
                for _ in actions:
                    _, reward = memory.attempt(phrase, action, actions, rng)
                    if reward > 0:
                        vectors.append(self.embed(phrase))
                        break
            self.examples[action] = vectors
            total = Counter()
            for vector in vectors:
                total.update(vector)
            self.centroids[action] = normalize_vector(total)

    @staticmethod
    def features(phrase):
        words = [re.sub(r"(?:으로|에서|[을를이에의가])$", "", word) for word in phrase.split()]
        chars = "".join(words)
        features = [f"word:{word}" for word in words] * 2
        for width in (1, 2, 3):
            features.extend(f"gram:{chars[index:index + width]}" for index in range(len(chars) - width + 1))
        return features

    def embed(self, phrase):
        features = self.features(phrase)
        activity = self.fly.propagate(features)[1:]
        graph = normalize_vector({index: value for index, value in enumerate(activity) if value})
        if not self.lexical:
            return graph
        lexical = normalize_vector({feature: count * math.log1p(len(PHRASES) / self.feature_classes[feature])
                                    for feature, count in Counter(features).items() if self.feature_classes[feature]})
        return {**{index: value * math.sqrt(0.3) for index, value in graph.items()},
                **{feature: value * math.sqrt(0.7) for feature, value in lexical.items()}}

    def ranks(self, phrase, *, prototypes=True):
        activity = self.embed(phrase)
        scores = {}
        for action, examples in self.examples.items():
            nearest = max(similarity(activity, example) for example in examples)
            scores[action] = (nearest + similarity(activity, self.centroids[action])) / 2 if prototypes else nearest
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)
