import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from fly import Fly
from lab import Experiment
from reward import RewardMemory


class CurriculumTest(unittest.TestCase):
    def test_training_uses_the_graph_to_distinguish_tokens(self):
        tokens = ["=", "("]
        fly = Fly(tokens)
        disconnected = Fly(tokens)
        disconnected.edges.clear()
        self.assertIsNone(fly.predict(fly.activity("speed □ 7")))

        examples = [("x □ 1", "="), ("show□x)", "(")]
        fly.train(examples)
        disconnected.train(examples)

        checks = ["speed □ 7", "print□speed)"]
        self.assertEqual([fly.predict(fly.activity(code)) for code in checks], tokens)
        self.assertEqual(disconnected.predict(disconnected.activity(checks[0])),
                         disconnected.predict(disconnected.activity(checks[1])))

    def test_reward_feedback_and_checkpoint_resume(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.json"
            experiment = Experiment(checkpoint)
            experiment.batch(practice=False)
            resumed = Experiment(checkpoint)

            self.assertGreater(resumed.state["positive"], 0)
            self.assertGreater(resumed.state["negative"], 0)
            self.assertEqual(resumed.state, experiment.state)
            resumed.batch(practice=False)
            self.assertEqual(resumed.state["batches"], 2)

    def test_bonus_is_capped_and_resets_after_success(self):
        memory = RewardMemory(None, [], [], {})
        rng = Mock()
        rng.choice.side_effect = lambda choices: choices[0]
        tokens = ("x", "y", "z", "w", "=")
        rewards = [memory.attempt("score □ 7", "=", tokens, rng)[1] for _ in range(6)]
        self.assertEqual(rewards, [-1, -1, -1, -1, 4, 1])

    def test_training_stops_only_after_every_check_is_correct(self):
        with TemporaryDirectory() as directory:
            experiment = Experiment(Path(directory) / "model.json", delay=0)
            self.assertFalse(experiment.mastered())
            for _ in range(260):
                if experiment.mastered():
                    break
                experiment.batch(practice=experiment.state["stage"] > len(experiment.lessons))
            self.assertTrue(experiment.mastered())
            self.assertFalse(experiment.start())
            self.assertGreater(experiment.state["bonus"], 0)


if __name__ == "__main__":
    unittest.main()
