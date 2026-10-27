#!/usr/bin/env python3
"""A population of separate learners that play each other (evolution strategies, one update per policy per generation).

Every policy is its own network. In each game one seat is the policy being perturbed; the other seats are other members of the
population (sampled from their current weights). Nothing but the game's own score is rewarded, nothing about style is coded:
whatever differences appear between the policies come from what works against the others.

    python train_pop.py --policies 6 --players 6 --generations 100 --out runs/pop1.npz
"""
from __future__ import annotations

import argparse
import collections
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ogsim.env import ACTION_NAMES, MultiEnv
from ogsim.policy import NumpyPolicy
from ogsim.world import Config


def play(job: dict) -> dict:
    """One game. Seat 0 = the candidate (weights job['theta']); other seats = job['others'] (list of weight vectors)."""
    n = 1 + len(job["others"])
    env = MultiEnv(Config(n_players=n, seed=job["seed"], systems=job["systems"]), decision_seconds=job["dt"], horizon_seconds=job["days"] * 86400)
    obs = env.reset()
    rng = np.random.default_rng(job["seed"])
    pols = [NumpyPolicy(job["hidden"], job["theta"])] + [NumpyPolicy(job["hidden"], t) for t in job["others"]]
    kinds = collections.Counter()
    done = False
    while not done:
        acts = {0: pols[0].act(env, 0, obs[0], rng=rng if job["explore"] > 0 else None, explore=job["explore"])}
        for i in range(1, n):
            acts[i] = pols[i].act(env, i, obs[i], rng=rng)
        kinds[ACTION_NAMES[acts[0][0]]] += 1
        obs, _, done, _ = env.step(acts)
    me = env.world.players[0]
    f = env.features(0)
    return {"score": me.score, "income": f["income"], "army": f["army"], "defence": f["defence"], "loot": me.looted, "lost": me.ships_lost,
            "planets": len(me.planets), "battles": len(me.battles), "kinds": dict(kinds),
            "rank": 1 + sum(q.score > me.score for q in env.world.players[1:])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=int, default=6)
    ap.add_argument("--players", type=int, default=6, help="players per game")
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--population", type=int, default=12, help="perturbations tried per policy per generation")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--explore", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=1800.0)
    ap.add_argument("--systems", type=int, default=60)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/pop.npz")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    base = NumpyPolicy(a.hidden)
    thetas = [base.init(rng) for _ in range(a.policies)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        for gen in range(a.generations):
            t0 = time.time()
            jobs, owner, eps_all = [], [], {}
            for i in range(a.policies):
                half = rng.normal(size=(a.population // 2, base.size))
                eps = np.concatenate([half, -half])
                eps_all[i] = eps
                for j, e in enumerate(eps):
                    others = [thetas[k] for k in rng.choice([k for k in range(a.policies) if k != i], a.players - 1, replace=True)]
                    jobs.append({"theta": thetas[i] + a.sigma * e, "others": others, "seed": 5000 + gen * 31 + j % (a.population // 2),
                                 "days": a.days, "hidden": a.hidden, "explore": a.explore, "dt": a.dt, "systems": a.systems})
                    owner.append(i)
            res = pool.map(play, jobs)
            stats = collections.defaultdict(list)
            for i in range(a.policies):
                mine = [r for r, o in zip(res, owner) if o == i]
                fit = np.array([r["score"] for r in mine])
                ranks = fit.argsort().argsort() / (len(fit) - 1) - 0.5
                thetas[i] = thetas[i] + a.lr / (len(fit) * a.sigma) * (eps_all[i].T @ ranks)
                stats[i] = mine
            line = []
            for i in range(a.policies):
                m = lambda k: np.mean([r[k] for r in stats[i]])
                line.append(f"p{i}: score {m('score'):6.0f} loot {m('loot'):6.0f} lost {m('lost'):6.0f} army {m('army'):7.0f} def {m('defence'):7.0f} pl {m('planets'):.1f}")
            print(f"gen {gen:3d} ({time.time() - t0:.0f}s)  " + " | ".join(line), flush=True)
            np.savez(a.out, thetas=np.array(thetas), hidden=a.hidden)


if __name__ == "__main__":
    main()
