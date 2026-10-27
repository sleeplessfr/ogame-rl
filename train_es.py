#!/usr/bin/env python3
"""Evolution-strategies training of a policy from nothing, in the fast simulator.

One learner plays against plain builder bots in a small universe; its fitness is how its final score compares with theirs.
    python train_es.py --generations 40 --population 24 --days 3 --out runs/es1.npz
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ogsim.bots import EcoBot, RandomBot
from ogsim.env import MultiEnv
from ogsim.policy import NumpyPolicy
from ogsim.world import Config

ARGS = None


def play(theta: np.ndarray, seed: int, days: float, opponents: int, hidden: int, dt: float = 1800.0, who: str = "eco") -> tuple[float, float]:
    """Returns (learner score, best opponent score) after the game."""
    n = 1 + opponents
    env = MultiEnv(Config(n_players=n, seed=seed, systems=60), decision_seconds=dt, horizon_seconds=days * 86400)
    obs = env.reset()
    pol = NumpyPolicy(hidden, theta)
    bots = [EcoBot(seed + i) if who == "eco" else RandomBot(seed + i) for i in range(n)]
    done = False
    while not done:
        acts = {0: pol.act(env, 0, obs[0])}
        for i in range(1, n):
            acts[i] = bots[i].act(env, i)
        obs, _, done, _ = env.step(acts)
    sc = [p.score for p in env.world.players]
    return sc[0], max(sc[1:])


def _job(args):
    return play(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--opponents", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/es.npz")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pol = NumpyPolicy(a.hidden)
    theta = pol.theta.copy()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        for gen in range(a.generations):
            t0 = time.time()
            world_seed = 1000 + gen
            eps = rng.normal(size=(a.population // 2, pol.size))
            eps = np.concatenate([eps, -eps])                               # antithetic pairs
            jobs = [(theta + a.sigma * e, world_seed, a.days, a.opponents, a.hidden) for e in eps]
            res = pool.map(_job, jobs)
            fit = np.array([me for me, best in res])                            # own final score (same world for everybody this generation)
            ranks = fit.argsort().argsort() / (len(fit) - 1) - 0.5
            theta = theta + a.lr / (a.population * a.sigma) * (eps.T @ ranks)
            me = np.mean([r[0] for r in res])
            print(f"gen {gen:3d}  learner score {me:8.1f}  best opponent {np.mean([r[1] for r in res]):8.1f}  "
                  f"best of population {max(r[0] for r in res):8.1f}  ({time.time() - t0:.1f}s)", flush=True)
            np.savez(a.out, theta=theta, hidden=a.hidden)


if __name__ == "__main__":
    main()
