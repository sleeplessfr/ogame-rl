#!/usr/bin/env python3
"""PPO self-play: one network plays every seat of many games; only the game's own score is rewarded.

    python train_ppo.py --players 12 --days 14 --dt 7200 --iters 200 --games 8 --out runs/ppo1.npz
"""
from __future__ import annotations

import argparse
import collections
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ogsim.env import ACTION_NAMES, MultiEnv
from ogsim.ppo import Net, gae
from ogsim.world import Config


def rollout(job: dict) -> dict:
    net = Net(job["hidden"])
    net.load(job["state"])
    n = job["players"]
    env = MultiEnv(Config(n_players=n, seed=job["seed"], systems=job["systems"]), decision_seconds=job["dt"], horizon_seconds=job["days"] * 86400)
    obs = env.reset()
    rng = np.random.default_rng(job["seed"])
    pids = list(range(n))
    recs, rews = [], []
    kinds = collections.Counter()
    done = False
    while not done:
        acts, rec = net.act(env, pids, obs, rng)
        for a in acts.values():
            kinds[ACTION_NAMES[a[0]]] += 1
        obs, r, done, _ = env.step(acts)
        recs.append(rec)
        rews.append(np.array([r[i] for i in pids]) / job["rscale"])
    T = len(recs)
    R = np.stack(rews)                       # T x n
    V = np.stack([x["v"] for x in recs])     # T x n
    ADV = np.zeros_like(R)
    RET = np.zeros_like(R)
    for s in range(n):
        ADV[:, s], RET[:, s] = gae(R[:, s], V[:, s], job["gamma"], job["lam"])
    batch = {k: np.concatenate([x[k] for x in recs]) for k in ("obs", "km", "am", "bm", "kind", "a", "b", "c", "logp")}
    batch["adv"] = ADV.reshape(-1)
    batch["ret"] = RET.reshape(-1)
    feats = [env.features(i) for i in pids]
    return {"batch": batch, "kinds": dict(kinds), "feats": feats, "score": [env.world.players[i].score for i in pids]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=12)
    ap.add_argument("--games", type=int, default=8, help="games per iteration")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--dt", type=float, default=7200.0)
    ap.add_argument("--systems", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--rscale", type=float, default=100.0, help="rewards are divided by this (only a unit change)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/ppo.npz")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    net = Net(args.hidden, args.seed)
    if args.resume:
        net.load(dict(np.load(args.resume)))
    rng = np.random.default_rng(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pool = Pool(args.workers)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        jobs = [dict(state=net.state(), hidden=args.hidden, players=args.players, seed=args.seed * 100000 + it * 100 + g, systems=args.systems,
                     dt=args.dt, days=args.days, gamma=args.gamma, lam=args.lam, rscale=args.rscale) for g in range(args.games)]
        res = pool.map(rollout, jobs)
        B = {k: np.concatenate([r["batch"][k] for r in res]) for k in res[0]["batch"]}
        adv = B["adv"]
        B["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        st = net.update(B, epochs=args.epochs, lr=args.lr, rng=rng, ent=args.ent)
        kinds = collections.Counter()
        for r in res:
            kinds.update(r["kinds"])
        tot = sum(kinds.values())
        fe = [f for r in res for f in r["feats"]]
        mean = lambda k: float(np.mean([f[k] for f in fe]))
        multi = float(np.mean([f["planets"] > 1 for f in fe]))
        print(f"it {it:3d} {time.time()-t0:6.0f}s score {mean('score'):8.1f} planets {mean('planets'):.2f} (>1: {multi:.0%}) loot {mean('loot'):9.0f} "
              f"army {mean('army'):9.0f} def {mean('defence'):9.0f} ent {st['ent']:.2f} kl {st['kl']:.3f} | "
              + " ".join(f"{k[:4]}{v/tot:.0%}" for k, v in kinds.most_common(5)), flush=True)
        if it % 5 == 0 or it == args.iters:
            np.savez(args.out, **net.state())
    pool.close()


if __name__ == "__main__":
    main()
