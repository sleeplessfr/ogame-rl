import unittest

import numpy as np

from ogsim.bots import RandomBot
from ogsim.env import MultiEnv, OBS_SIZE
from ogsim.ppo import Net, gae
from ogsim.world import Config


def fake_batch(net, n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, OBS_SIZE)) * 0.3
    km = rng.random((n, 9)) < 0.6
    km[:, 0] = True
    kind = np.array([rng.choice(np.flatnonzero(km[i])) for i in range(n)])
    am = rng.random((n, 6)) < 0.7
    am[:, 0] = True
    bm = rng.random((n, 25)) < 0.5
    bm[:, 0] = True
    a = np.array([rng.choice(np.flatnonzero(am[i])) if kind[i] else 0 for i in range(n)])
    b = np.array([rng.choice(np.flatnonzero(bm[i])) if kind[i] else 0 for i in range(n)])
    c = rng.integers(0, 5, n) * (kind > 0)
    h = net.trunk(X)
    # old log-probs a bit different from the current ones so that the ratio is not 1
    B = {"obs": X, "km": km, "am": am, "bm": bm, "kind": kind, "a": a, "b": b, "c": c, "logp": np.zeros(n) - 2.0,
         "adv": rng.normal(size=n), "ret": rng.normal(size=n)}
    return B


class PPOTest(unittest.TestCase):
    def test_gradient_matches_finite_differences(self):
        net = Net(hidden=8, seed=1)
        for k in net.p:    # make the heads non-trivial
            net.p[k] = net.p[k] + np.random.default_rng(3).normal(0, 0.2, net.p[k].shape)
        B = fake_batch(net)
        _, G, _ = net.loss_grad(B, clip=10.0)    # a wide clip keeps every sample on the differentiable branch
        rng = np.random.default_rng(5)
        for name in ("W1", "Wk", "Wa", "Wb", "Wc", "wv", "bk", "bb"):
            arr = net.p[name]
            for _ in range(4):
                ix = tuple(rng.integers(0, s) for s in arr.shape)
                old = arr[ix]
                eps = 1e-5
                arr[ix] = old + eps
                lp = net.loss_grad(B, clip=10.0)[0]
                arr[ix] = old - eps
                lm = net.loss_grad(B, clip=10.0)[0]
                arr[ix] = old
                num = (lp - lm) / (2 * eps)
                self.assertAlmostEqual(G[name][ix], num, delta=1e-6 + 1e-4 * abs(num), msg=f"{name}{ix}")

    def test_gae_matches_hand_calculation(self):
        adv, ret = gae(np.array([1.0, 0.0, 2.0]), np.array([0.5, 0.5, 0.5]), gamma=0.9, lam=1.0)
        d2 = 2 - 0.5
        d1 = 0 + 0.9 * 0.5 - 0.5
        d0 = 1 + 0.9 * 0.5 - 0.5
        self.assertAlmostEqual(adv[2], d2)
        self.assertAlmostEqual(adv[1], d1 + 0.9 * d2)
        self.assertAlmostEqual(adv[0], d0 + 0.9 * (d1 + 0.9 * d2))
        np.testing.assert_allclose(ret, adv + 0.5)

    def test_acts_legally_in_the_environment(self):
        env = MultiEnv(Config(n_players=5, seed=2, systems=40), decision_seconds=1800, horizon_seconds=86400)
        obs = env.reset()
        net = Net(32, 0)
        rng = np.random.default_rng(0)
        bad = total = 0
        done = False
        while not done:
            acts, rec = net.act(env, list(range(5)), obs, rng)
            obs, _, done, info = env.step(acts)
            for pid, ok in info["valid"].items():
                total += 1
                bad += (not ok and acts[pid][0] in (1, 2, 3))
            self.assertEqual(rec["obs"].shape[1], OBS_SIZE)
        self.assertEqual(bad, 0)

    def test_update_improves_the_objective_on_a_fixed_batch(self):
        net = Net(16, 0)
        B = fake_batch(net, 200)
        h_logp = None
        before = net.loss_grad(B)[2]["loss"]
        net.update(B, epochs=30, minibatch=200, lr=1e-2)
        after = net.loss_grad(B)[2]["loss"]
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
