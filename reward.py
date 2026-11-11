import math


class RewardMemory:
    """Store the feedback for attempted tokens and reuse it for similar graph activations."""

    def __init__(self, fly, training, checks, feedback, *, project=None):
        self.feedback = feedback
        prototypes = {prompt: self.embedding(fly, prompt, project) for _, prompt, _ in training}
        self.neighbors = {}
        for _, prompt, _ in checks:
            activity = self.embedding(fly, prompt, project)
            self.neighbors[prompt] = sorted(prototypes, key=lambda candidate: -sum(
                value * activity.get(index, 0) for index, value in prototypes[candidate].items()))

    @staticmethod
    def embedding(fly, prompt, project=None):
        activity = (project(prompt) if project else fly.activity(prompt))[1:]
        scale = math.hypot(*activity)
        return {index: value / scale for index, value in enumerate(activity) if value}

    def predict(self, prompt):
        for candidate in self.neighbors[prompt]:
            scores = self.feedback.get(candidate, {}).get("scores", {})
            if scores:
                best = max(scores, key=scores.get)
                if scores[best] > 0:
                    return best
        return None

    def priority(self, prompt):
        return 1 + min(4, self.feedback.get(prompt, {}).get("streak", 0))

    def attempt(self, prompt, expected, tokens, rng):
        feedback = self.feedback.setdefault(prompt, {"scores": {}, "streak": 0})
        scores = feedback["scores"]
        best = max(scores, key=scores.get) if scores else None
        if best is not None and scores[best] > 0:
            chosen = best
        else:
            chosen = rng.choice([token for token in tokens if token not in scores] or tokens)
        if chosen == expected:
            reward = 1 + min(3, feedback["streak"])
            feedback["streak"] = 0
        else:
            reward = -1
            feedback["streak"] += 1
        scores[chosen] = scores.get(chosen, 0) + reward
        return chosen, reward
