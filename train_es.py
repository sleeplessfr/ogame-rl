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
from ogsim.env import PERSONALITIES, MultiEnv
from ogsim.policy import NumpyPolicy
from ogsim.world import Config

ARGS = None


def play(theta: np.ndarray, seed: int, days: float, opponents: int, hidden: int, personality: str = "raider", opp: str = "self",
         mean_theta: np.ndarray | None = None, dt: float = 1800.0, explore: float = 0.0) -> dict:
    """One game. Player 0 is the learner (reward = its personality); the others are copies of the current mean policy
    (opp='self', sampled so they vary), plain builder bots (opp='eco') or random bots (opp='random')."""
    n = 1 + opponents
    pers = {i: PERSONALITIES[personality] for i in range(n)}
    env = MultiEnv(Config(n_players=n, seed=seed, systems=60), decision_seconds=dt, horizon_seconds=days * 86400, personalities=pers)
    obs = env.reset()
    pol = NumpyPolicy(hidden, theta)
    rng = np.random.default_rng(seed)
    bots = [EcoBot(seed + i) if opp == "eco" else RandomBot(seed + i) for i in range(n)]
    mean_pol = NumpyPolicy(hidden, mean_theta if mean_theta is not None else theta)
    total, done = 0.0, False
    while not done:
        acts = {0: pol.act(env, 0, obs[0], rng=rng if explore > 0 else None, explore=explore)}
        for i in range(1, n):
            acts[i] = mean_pol.act(env, i, obs[i], rng=rng) if opp == "self" else bots[i].act(env, i)
        obs, rew, done, _ = env.step(acts)
        total += rew[0]
    me = env.world.players[0]
    feats = env.features(0)
    rank = 1 + sum(q.score > me.score for q in env.world.players[1:])
    return {"reward": total, "score": me.score, "income": feats["income"], "army": feats["army"], "loot": me.looted,
            "planets": len(me.planets), "battles": len(me.battles), "rank": rank}


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
    ap.add_argument("--personality", default="raider", choices=sorted(PERSONALITIES))
    ap.add_argument("--opp", default="self", choices=("self", "eco", "random"))
    ap.add_argument("--explore", type=float, default=0.0, help="chance per step that the learner tries a random legal kind of action")
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
            jobs = [(theta + a.sigma * e, world_seed, a.days, a.opponents, a.hidden, a.personality, a.opp, theta, 1800.0, a.explore) for e in eps]
            res = pool.map(_job, jobs)
            fit = np.array([r["reward"] for r in res])                       # the personality's own reward (same world for everybody this generation)
            ranks = fit.argsort().argsort() / (len(fit) - 1) - 0.5
            theta = theta + a.lr / (a.population * a.sigma) * (eps.T @ ranks)
            m = lambda k: np.mean([r[k] for r in res])
            print(f"gen {gen:3d}  reward {m('reward'):6.2f}  score {m('score'):8.0f}  income/h {m('income'):9.0f}  army {m('army'):9.0f}  "
                  f"loot {m('loot'):8.0f}  planets {m('planets'):.1f}  battles {m('battles'):5.1f}  rank {m('rank'):.2f}  ({time.time() - t0:.1f}s)", flush=True)
            np.savez(a.out, theta=theta, hidden=a.hidden)


if __name__ == "__main__":
    main()
