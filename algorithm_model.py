import math
import random
import re
from collections import Counter

from algorithm_phrases import training_phrases
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
        phrases_by_action = training_phrases()
        self.feature_classes = Counter(
            feature for phrases in phrases_by_action.values()
            for feature in {value for phrase in phrases for value in self.features(phrase)}
        )
        self.phrase_counts = {action: len(phrases) for action, phrases in phrases_by_action.items()}
        self.feature_counts = {action: Counter(feature for phrase in phrases for feature in set(self.features(phrase)))
                               for action, phrases in phrases_by_action.items()}
        self.total_features = sum(self.feature_counts.values(), Counter())
        self.total_phrases = sum(self.phrase_counts.values())
        memory = RewardMemory(fly, [], [], {})
        rng = random.Random(42)
        actions = []
        for action, phrases in phrases_by_action.items():
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
        lexical = normalize_vector({feature: count * math.log1p(len(self.phrase_counts) / self.feature_classes[feature])
                                    for feature, count in Counter(features).items() if self.feature_classes[feature]})
        return {**{index: value * math.sqrt(0.3) for index, value in graph.items()},
                **{feature: value * math.sqrt(0.7) for feature, value in lexical.items()}}

    def contrast(self, action, features):
        in_class = self.phrase_counts[action] + 1
        outside = self.total_phrases - self.phrase_counts[action] + 1
        class_features = self.feature_counts[action]
        evidence = 0.0
        for feature in features:
            present = class_features[feature]
            positive = (present + 0.5) / in_class
            negative = (self.total_features[feature] - present + 0.5) / outside
            evidence += math.log(positive / negative)
        return evidence / max(1, len(features))

    def ranks(self, phrase, *, prototypes=True, contrast=True):
        activity = self.embed(phrase)
        features = [feature for feature in dict.fromkeys(self.features(phrase)) if self.total_features[feature]]
        scores = {}
        for action, examples in self.examples.items():
            nearest = max(similarity(activity, example) for example in examples)
            scores[action] = (nearest + similarity(activity, self.centroids[action])) / 2 if prototypes else nearest
            if self.lexical and contrast:
                scores[action] += 0.3 * self.contrast(action, features)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)
