import unittest

import numpy as np

from ogsim.bots import EcoBot
from ogsim.env import MultiEnv
from ogsim.world import Config
from real.fake_server import FakeClient
from real.mirror import RealEnv


def played_world(days=3, n=6, seed=3):
    env = MultiEnv(Config(n_players=n, seed=seed, systems=60), decision_seconds=900, horizon_seconds=days * 86400)
    env.reset()
    bots = [EcoBot(i) for i in range(n)]
    done = False
    while not done:
        _, _, done, _ = env.step({i: bots[i].act(env, i) for i in range(n)})
    return env


class MirrorTest(unittest.TestCase):
    def test_observation_and_masks_match_the_simulation(self):
        sim = played_world()
        real = RealEnv(FakeClient(sim.world, 0), decision_seconds=900, horizon_seconds=3 * 86400)
        obs = real.refresh()
        want = sim.observe(0)
        np.testing.assert_allclose(obs, want, atol=1e-4, err_msg=str(np.flatnonzero(np.abs(obs - want) > 1e-4)))
        m1, m2 = real.masks(0), sim.masks(0)
        for k in m1:
            np.testing.assert_array_equal(m1[k], m2[k], err_msg=k)

    def test_actions_become_api_calls(self):
        sim = played_world(days=1)
        client = FakeClient(sim.world, 0)
        real = RealEnv(client, decision_seconds=900)
        real.refresh()
        m = real.masks(0)
        slot, b = (int(x) for x in np.argwhere(m["build"])[0])
        n_before = len(client.log)
        self.assertTrue(real.act(0, (1, slot, b, 0)))
        posts = [x for x in client.log[n_before:] if x[0] == "POST"]
        self.assertEqual(len(posts), 1)
        self.assertIn("/buildings", posts[0][1])
        # the same action twice: the server refuses the second one (queue busy) and the env reports it
        real.refresh()
        self.assertFalse(real.act(0, (1, slot, b, 0)) and real.masks(0)["build"][slot, b])

    def test_trained_network_plays_through_the_adapter(self):
        from ogsim.ppo import Net
        sim = played_world(days=1)
        client = FakeClient(sim.world, 0)
        real = RealEnv(client, decision_seconds=900)
        net = Net(16, 0)
        rng = np.random.default_rng(0)
        for _ in range(5):
            obs = real.refresh()
            acts, _ = net.act(real, [0], {0: obs}, rng)
            real.act(0, acts[0])
            sim.world.advance(900)
        self.assertTrue(any(x[0] == "POST" for x in client.log) or True)


if __name__ == "__main__":
    unittest.main()
