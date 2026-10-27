#!/usr/bin/env python3
"""Plays saved policies in one big universe (like the real server) and prints what each did.
    python eval_pop.py --weights runs/big1.npz --players 200 --systems 499 --days 14
Seats are filled with the saved policies in turn."""
import argparse, collections, time
import numpy as np
from ogsim.env import ACTION_NAMES, MultiEnv
from ogsim.policy import NumpyPolicy
from ogsim.world import Config

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True); ap.add_argument("--players", type=int, default=200); ap.add_argument("--systems", type=int, default=499)
ap.add_argument("--days", type=float, default=14); ap.add_argument("--dt", type=float, default=3600); ap.add_argument("--seed", type=int, default=99)
a = ap.parse_args()
d = np.load(a.weights); th = d["thetas"]; hidden = int(d["hidden"]); P = len(th)
pols = [NumpyPolicy(hidden, t) for t in th]
env = MultiEnv(Config(n_players=a.players, seed=a.seed, systems=a.systems), decision_seconds=a.dt, horizon_seconds=a.days * 86400)
obs = env.reset(); rng = np.random.default_rng(a.seed)
kinds = [collections.Counter() for _ in range(P)]
t0 = time.time(); done = False
while not done:
    acts = {i: pols[i % P].act(env, i, obs[i], rng=rng) for i in range(a.players)}
    for i, ac in acts.items():
        kinds[i % P][ACTION_NAMES[ac[0]]] += 1
    obs, _, done, _ = env.step(acts)
print(f"{a.players} players, {a.days:.0f} game days in {time.time() - t0:.0f}s")
by = collections.defaultdict(list)
for i, p in enumerate(env.world.players):
    by[i % P].append(p)
for k in range(P):
    ps = by[k]
    tot = sum(kinds[k].values()) or 1
    print(f"policy {k}: n={len(ps)} score avg {np.mean([p.score for p in ps]):9.0f}  planets {np.mean([len(p.planets) for p in ps]):.2f}  "
          f"loot {np.mean([p.looted for p in ps]):10.0f}  lost {np.mean([p.ships_lost for p in ps]):10.0f}  " +
          " ".join(f"{n}:{kinds[k][n] / tot:.0%}" for n in ACTION_NAMES if kinds[k][n]))
sc = sorted(((p.score, i % P) for i, p in enumerate(env.world.players)), reverse=True)
print("top 10 by score (score, policy):", [(round(s), k) for s, k in sc[:10]])
