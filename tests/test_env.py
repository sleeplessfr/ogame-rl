import time
import unittest

import numpy as np

from ogsim.bots import EcoBot, RandomBot
from ogsim.env import ARGS, BUILDINGS, OBS_SIZE, MultiEnv
from ogsim.world import Config


class EnvTest(unittest.TestCase):
    def make(self, n=6):
        return MultiEnv(Config(n_players=n, seed=2, systems=40), decision_seconds=900, horizon_seconds=86400)

    def test_reset_and_obs_shape(self):
        env = self.make()
        obs = env.reset()
        self.assertEqual(len(obs), 6)
        self.assertEqual(obs[0].shape, (OBS_SIZE,))
        self.assertTrue(np.isfinite(obs[0]).all())

    def test_illegal_action_does_nothing_and_costs_a_little(self):
        env = self.make()
        env.reset()
        obs, rew, done, info = env.step({0: (1, 0, BUILDINGS.index("fusion_plant"), 0)})
        self.assertFalse(info["valid"][0])
        self.assertLess(rew[0], 0)

    def test_building_is_rewarded_with_score(self):
        env = self.make()
        env.reset()
        _, rew, _, info = env.step({0: (1, 0, BUILDINGS.index("solar_plant"), 0)})
        self.assertTrue(info["valid"][0])
        self.assertGreater(rew[0], 0)

    def test_mask_matches_the_game(self):
        env = self.make()
        env.reset()
        m = env.masks(0)
        for slot, b in np.argwhere(m["build"]).tolist():
            self.assertTrue(env.act(0, (1, slot, b, 0)))
            break
        self.assertFalse(env.masks(0)["build"].any())   # queue busy now

    def test_random_bots_play_a_day_without_errors(self):
        env = self.make(8)
        env.reset()
        bots = [RandomBot(i) for i in range(8)]
        done = False
        while not done:
            _, _, done, _ = env.step({i: bots[i].act(env, i) for i in range(8)})
        self.assertTrue(any(p.score > 0 for p in env.world.players))

    def test_masked_actions_are_always_valid(self):
        env = self.make(4)
        env.reset()
        bots = [RandomBot(i, idle=0.0) for i in range(4)]
        bad = total = 0
        for _ in range(60):
            acts = {i: bots[i].act(env, i) for i in range(4)}
            _, _, _, info = env.step(acts)
            for i, ok in info["valid"].items():
                total += 1
                bad += (not ok and acts[i][0] in (1, 2, 3))
        self.assertEqual(bad, 0, f"{bad}/{total} masked builds/research/units were refused")

    def test_ecobot_grows(self):
        env = MultiEnv(Config(n_players=4, seed=4, systems=40), decision_seconds=900, horizon_seconds=3 * 86400)
        env.reset()
        bots = [EcoBot(i) for i in range(4)]
        done = False
        while not done:
            _, _, done, _ = env.step({i: bots[i].act(env, i) for i in range(4)})
        self.assertGreater(env.world.players[0].score, 20)

    def test_speed(self):
        env = MultiEnv(Config(n_players=16, seed=5), decision_seconds=900, horizon_seconds=2 * 86400)
        env.reset()
        bots = [RandomBot(i) for i in range(16)]
        t0, steps = time.time(), 0
        done = False
        while not done:
            _, _, done, _ = env.step({i: bots[i].act(env, i) for i in range(16)})
            steps += 16
        rate = steps / (time.time() - t0)
        print(f"\n{rate:,.0f} player-steps per second (16 random players, 900 s steps)")
        self.assertGreater(rate, 500)


if __name__ == "__main__":
    unittest.main()
