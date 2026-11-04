import unittest
import tempfile
from pathlib import Path

import numpy as np

from platformer.agents import RandomPolicy
from platformer.env import DIFFICULTY_CONFIGS, PlatformerEnv
from platformer.evaluation import evaluate_policy, generalization_gap, steps_to_threshold
from platformer.visualize import create_episode_visualization


class PlatformerEnvironmentTests(unittest.TestCase):
    def test_seeded_reset_and_transitions_are_reproducible(self) -> None:
        first = PlatformerEnv()
        second = PlatformerEnv()
        first_obs, _ = first.reset(seed=73)
        second_obs, _ = second.reset(seed=73)
        np.testing.assert_array_equal(first_obs, second_obs)
        self.assertEqual(first.platforms, second.platforms)
        for action in (2, 2, 5, 2, 7):
            first_result = first.step(action)
            second_result = second.step(action)
            np.testing.assert_array_equal(first_result[0], second_result[0])
            self.assertEqual(first_result[1:], second_result[1:])

    def test_unseeded_reset_advances_level_rng(self) -> None:
        env = PlatformerEnv()
        env.reset(seed=73)
        first_level = list(env.platforms)
        env.reset()
        self.assertNotEqual(first_level, env.platforms)

    def test_difficulty_levels_are_paired_and_progressively_structured(self) -> None:
        profiles = {difficulty: [] for difficulty in DIFFICULTY_CONFIGS}
        for difficulty in DIFFICULTY_CONFIGS:
            for seed in range(100):
                env = PlatformerEnv(difficulty=difficulty)
                _, info = env.reset(seed=seed)
                profiles[difficulty].append(info["level_profile"])
                self.assertEqual(info["difficulty"], difficulty)

            first = PlatformerEnv(difficulty=difficulty)
            second = PlatformerEnv(difficulty=difficulty)
            _, first_info = first.reset(seed=123)
            _, second_info = second.reset(seed=123)
            self.assertEqual(first_info["level_profile"], second_info["level_profile"])
            self.assertEqual(first.platforms, second.platforms)

        mean_gaps = {
            difficulty: np.mean([profile["mean_gap"] for profile in rows])
            for difficulty, rows in profiles.items()
        }
        mean_widths = {
            difficulty: np.mean([profile["mean_platform_width"] for profile in rows])
            for difficulty, rows in profiles.items()
        }
        self.assertLess(mean_gaps["easy"], mean_gaps["medium"])
        self.assertLess(mean_gaps["medium"], mean_gaps["hard"])
        self.assertGreater(mean_widths["easy"], mean_widths["medium"])
        self.assertGreater(mean_widths["medium"], mean_widths["hard"])
        env = PlatformerEnv(difficulty="easy")
        _, info = env.reset(seed=123, options={"difficulty": "hard"})
        self.assertEqual(info["difficulty"], "hard")
        self.assertEqual(info["level_profile"]["difficulty"], "hard")

    def test_moving_heavy_layout_is_a_distinct_repeatable_holdout(self) -> None:
        standard = PlatformerEnv(difficulty="medium", layout_variant="standard")
        moving_heavy = PlatformerEnv(
            difficulty="medium", layout_variant="moving_heavy"
        )
        _, standard_info = standard.reset(seed=81)
        _, moving_info = moving_heavy.reset(seed=81)
        self.assertEqual(moving_info["layout_variant"], "moving_heavy")
        self.assertGreater(
            moving_info["level_profile"]["moving_platforms"],
            standard_info["level_profile"]["moving_platforms"],
        )
        self.assertGreater(
            moving_info["level_profile"]["mean_moving_amplitude"],
            standard_info["level_profile"]["mean_moving_amplitude"],
        )
        for field in ("mean_gap", "max_gap", "mean_platform_width", "hazards", "enemies"):
            self.assertEqual(
                standard_info["level_profile"][field],
                moving_info["level_profile"][field],
            )
        second = PlatformerEnv(difficulty="medium", layout_variant="moving_heavy")
        _, second_info = second.reset(seed=81)
        self.assertEqual(moving_info["level_profile"], second_info["level_profile"])

    def test_obstacle_switches_and_death_reason_are_reported(self) -> None:
        env = PlatformerEnv(
            difficulty="hard", hazards_enabled=False, enemies_enabled=False
        )
        env.reset(seed=44)
        self.assertEqual(env.hazards, [])
        self.assertEqual(env.enemies, [])
        _, _, _, _, info = env.step(0)
        self.assertIn(info["death_reason"], (None, "fall", "hazard", "enemy"))

    def test_fall_death_has_specific_diagnostic_reason(self) -> None:
        env = PlatformerEnv()
        env.reset(seed=44)
        env.y = -8.1
        _, _, terminated, _, info = env.step(0)
        self.assertTrue(terminated)
        self.assertTrue(info["died"])
        self.assertEqual(info["death_reason"], "fall")

    def test_observation_and_actions_match_declared_spaces(self) -> None:
        env = PlatformerEnv()
        observation, _ = env.reset(seed=5)
        self.assertEqual(observation.shape, (42,))
        self.assertTrue(env.observation_space.contains(observation))
        with self.assertRaises(ValueError):
            env.step(env.action_space.n)

    def test_step_after_episode_end_requires_reset(self) -> None:
        env = PlatformerEnv(max_episode_steps=1)
        env.reset(seed=4)
        _, _, terminated, truncated, _ = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            env.step(0)


class AgentAndMetricsTests(unittest.TestCase):
    def test_random_evaluation_and_summary_metrics(self) -> None:
        env = PlatformerEnv(max_episode_steps=8)
        first = evaluate_policy(env, RandomPolicy(seed=1), seeds=[3, 4])
        second = evaluate_policy(env, RandomPolicy(seed=1), seeds=[3, 4])
        self.assertEqual(first, second)
        self.assertEqual(len(first["episodes"]), 2)
        self.assertIn("completion_rate", first)
        self.assertEqual(steps_to_threshold([(10, 0.4), (20, 0.8)]), 20)
        self.assertIsNone(steps_to_threshold([(10, 0.4)]))
        self.assertAlmostEqual(generalization_gap(0.8, 0.5), 0.3)

    def test_visualization_writes_replayable_html(self) -> None:
        env = PlatformerEnv(max_episode_steps=3)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "episode.html"
            summary = create_episode_visualization(
                env,
                RandomPolicy(seed=1),
                seed=9,
                output_path=output,
                algorithm="random",
            )
            page = output.read_text(encoding="utf-8")
        self.assertEqual(summary["steps"], 3)
        self.assertIn("const frames =", page)
        self.assertIn("id=\"scrubber\"", page)
        self.assertIn("Gems", page)


if __name__ == "__main__":
    unittest.main()
