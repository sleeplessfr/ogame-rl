#!/usr/bin/env python3
"""PPO with a population: K separate networks share each game; every seat is played by one of them (reshuffled every game).
Each network learns only from its own seats. Only the game's own score is rewarded, nothing about style is coded.

    python train_ppo_pop.py --policies 4 --players 12 --days 14 --dt 600 --iters 100 --out runs/pop_ppo.npz
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
    K, n = len(job["states"]), job["players"]
    nets = []
    for st in job["states"]:
        net = Net(job["hidden"])
        net.load(st)
        nets.append(net)
    rng = np.random.default_rng(job["seed"])
    owner = rng.permutation([i % K for i in range(n)])          # seat -> policy, new every game
    groups = [[s for s in range(n) if owner[s] == k] for k in range(K)]
    env = MultiEnv(Config(n_players=n, seed=job["seed"], systems=job["systems"]), decision_seconds=job["dt"], horizon_seconds=job["days"] * 86400)
    obs = env.reset()
    recs = [[] for _ in range(K)]
    rews = []
    kinds = [collections.Counter() for _ in range(K)]
    done = False
    while not done:
        acts = {}
        for k in range(K):
            if groups[k]:
                a, rec = nets[k].act(env, groups[k], obs, rng)
                acts.update(a)
                recs[k].append(rec)
                for x in a.values():
                    kinds[k][ACTION_NAMES[x[0]]] += 1
        obs, r, done, _ = env.step(acts)
        rews.append(np.array([r[i] for i in range(n)]) / job["rscale"])
    R = np.stack(rews)
    out = []
    for k in range(K):
        if not groups[k]:
            out.append(None)
            continue
        Rk = R[:, groups[k]]
        Vk = np.stack([x["v"] for x in recs[k]])
        ADV, RET = np.zeros_like(Rk), np.zeros_like(Rk)
        for s in range(Rk.shape[1]):
            ADV[:, s], RET[:, s] = gae(Rk[:, s], Vk[:, s], job["gamma"], job["lam"])
        b = {key: np.concatenate([x[key] for x in recs[k]]) for key in ("obs", "km", "am", "bm", "kind", "a", "b", "c", "logp")}
        b["adv"], b["ret"] = ADV.reshape(-1), RET.reshape(-1)
        out.append(b)
    feats = [[env.features(s) for s in groups[k]] for k in range(K)]
    return {"batches": out, "kinds": [dict(c) for c in kinds], "feats": feats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", type=int, default=4)
    ap.add_argument("--players", type=int, default=12)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--systems", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.03)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--rscale", type=float, default=1000.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/pop_ppo.npz")
    ap.add_argument("--resume", default=None, help="a single-network .npz to start every policy from (they then diverge)")
    args = ap.parse_args()
    K = args.policies
    nets = [Net(args.hidden, args.seed + k) for k in range(K)]
    if args.resume:
        st = dict(np.load(args.resume))
        if "p0_W1" in st:      # a population file: network k continues from its own weights
            for k, n in enumerate(nets):
                n.load({key[len(f"p{k}_"):]: v for key, v in st.items() if key.startswith(f"p{k}_")})
        else:
            for n in nets:
                n.load(st)
    rng = np.random.default_rng(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pool = Pool(args.workers)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        jobs = [dict(states=[n.state() for n in nets], hidden=args.hidden, players=args.players, seed=args.seed * 100000 + it * 100 + g,
                     systems=args.systems, dt=args.dt, days=args.days, gamma=args.gamma, lam=args.lam, rscale=args.rscale) for g in range(args.games)]
        res = pool.map(rollout, jobs)
        line = []
        for k in range(K):
            bs = [r["batches"][k] for r in res if r["batches"][k] is not None]
            if not bs:
                continue
            B = {key: np.concatenate([b[key] for b in bs]) for key in bs[0]}
            B["adv"] = (B["adv"] - B["adv"].mean()) / (B["adv"].std() + 1e-8)
            nets[k].update(B, epochs=args.epochs, lr=args.lr, rng=rng, ent=args.ent)
            fe = [f for r in res for f in r["feats"][k]]
            kc = collections.Counter()
            for r in res:
                kc.update(r["kinds"][k])
            tot = max(1, sum(kc.values()))
            m = lambda key: float(np.mean([f[key] for f in fe]))
            line.append(f"[{k}] sc {m('score'):7.0f} pl {m('planets'):.1f} lo {m('loot'):6.0f} " + "/".join(f"{a[:2]}{v/tot:.0%}" for a, v in kc.most_common(3)))
        print(f"it {it:3d} {time.time()-t0:6.0f}s " + " | ".join(line), flush=True)
        if it % 5 == 0 or it == args.iters:
            np.savez(args.out, **{f"p{k}_{key}": v for k, n in enumerate(nets) for key, v in n.state().items()})
    pool.close()


if __name__ == "__main__":
    main()
