#!/usr/bin/env python3
"""Let trained agents play on a real OGameX server (or, with --fake, on an in-process stand-in driven by the simulator).

    python run_real.py --base-url http://localhost/api/v1 --weights runs/ppo4.npz --agents 20 --dt 600
    python run_real.py --fake --weights runs/ppo4.npz --agents 8 --days 2          # no server needed: checks the whole loop

Each agent is its own account (agent001@agents.local ...). Every `dt` seconds an agent reads the server, builds the same observation
as in training, picks one action with the network and sends it through the API. The network never sees anything but the game.
`--weights` can be given several times: agent k uses file k mod (number of files), like a population of separate policies.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from ogsim.ppo import Net
from real.api import ApiClient, ApiError
from real.mirror import RealEnv


def load_nets(path: str) -> list[Net]:
    """One network from a plain .npz, or every network of a population file (keys p0_W1, p1_W1, ...)."""
    st = dict(np.load(path))
    groups = sorted({k.split("_", 1)[0] for k in st if k[0] == "p" and "_" in k and k.split("_", 1)[0][1:].isdigit()}, key=lambda g: int(g[1:]))
    if not groups:
        groups = [""]
    nets = []
    for g in groups:
        sub = {k[len(g) + 1:] if g else k: v for k, v in st.items() if (k.startswith(g + "_") if g else True)}
        net = Net(sub["W1"].shape[1])
        net.load(sub)
        nets.append(net)
    return nets


def login(base: str, email: str, password: str, tokens: dict, path: Path) -> ApiClient:
    if email in tokens:
        return ApiClient(base, tokens[email])
    anon = ApiClient(base)
    try:
        data = anon.post("/auth/register", {"email": email, "password": password, "token_name": "agent"})
    except ApiError:
        data = anon.post("/auth/login", {"email": email, "password": password, "token_name": "agent"})
    tokens[email] = data["token"]
    path.write_text(json.dumps(tokens))
    return ApiClient(base, data["token"])


class Highscore:
    """One highscore read shared by all agents of this process (refreshed at most every `ttl` seconds)."""
    def __init__(self, ttl: float = 120.0):
        self.ttl, self.at, self.rows, self.lock = ttl, 0.0, [], threading.Lock()

    def get(self, env: RealEnv) -> list[dict]:
        with self.lock:
            if time.monotonic() - self.at > self.ttl or not self.rows:
                self.rows, self.at = env.fetch_highscore(), time.monotonic()
            return self.rows


def agent_loop(env: RealEnv, net: Net, hs: Highscore, stop: threading.Event, dt: float, seed: int, stats: dict, key: str) -> None:
    rng = np.random.default_rng(seed)
    while not stop.is_set():
        t0 = time.monotonic()
        try:
            obs = env.refresh(hs.get(env))
            acts, _ = net.act(env, [0], {0: obs}, rng)
            ok = env.act(0, acts[0])
            stats[key] = {"action": acts[0], "ok": ok, "score": env.world.players[0].score, "planets": len(env.world.players[0].planets)}
        except ApiError as e:
            stats[key] = {"error": str(e)}
            print(time.strftime("%H:%M:%S"), key, "API error:", e, flush=True)
            if e.code == "rate_limited":
                stop.wait(e.details.get("retry_after", 20))
        except Exception as e:      # never let one agent die silently
            import traceback
            stats[key] = {"error": repr(e)}
            print(time.strftime("%H:%M:%S"), key, "ERROR:", repr(e), flush=True)
            traceback.print_exc()
        stop.wait(max(0.0, dt - (time.monotonic() - t0)))


def run_fake(args, nets) -> None:
    from ogsim.env import MultiEnv
    from ogsim.world import Config
    from real.fake_server import FakeClient
    sim = MultiEnv(Config(n_players=args.agents, seed=args.seed, systems=args.systems), decision_seconds=args.dt)
    sim.reset()
    envs = [RealEnv(FakeClient(sim.world, i), decision_seconds=args.dt) for i in range(args.agents)]
    rng = np.random.default_rng(args.seed)
    for step in range(int(args.days * 86400 / args.dt)):
        for i, e in enumerate(envs):
            obs = e.refresh()
            acts, _ = nets[i % len(nets)].act(e, [0], {0: obs}, rng)
            e.act(0, acts[0])
        sim.world.advance(args.dt)
        if step % 24 == 0:
            sc = [p.score for p in sim.world.players]
            print(f"t={sim.world.t/86400:5.2f}d  mean score {np.mean(sc):9.1f}  best {max(sc):9.1f}  planets {np.mean([len(p.planets) for p in sim.world.players]):.2f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost/api/v1")
    ap.add_argument("--weights", action="append", required=True)
    ap.add_argument("--agents", type=int, default=10)
    ap.add_argument("--dt", type=float, default=600.0, help="seconds between two decisions of an agent (training used 600)")
    ap.add_argument("--password", default="agent-password-123")
    ap.add_argument("--tokens", default="runs/tokens.json")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--days", type=float, default=2.0, help="(--fake) game days to play")
    ap.add_argument("--systems", type=int, default=60)
    args = ap.parse_args()
    nets = [n for w in args.weights for n in load_nets(w)]
    if args.fake:
        return run_fake(args, nets)
    tpath = Path(args.tokens)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tokens = json.loads(tpath.read_text()) if tpath.exists() else {}
    hs, stop, stats = Highscore(), threading.Event(), {}
    threads = []
    for i in range(args.agents):
        email = f"agent{i + 1:03d}@agents.local"
        client = login(args.base_url, email, args.password, tokens, tpath)
        env = RealEnv(client, decision_seconds=args.dt)
        th = threading.Thread(target=agent_loop, args=(env, nets[i % len(nets)], hs, stop, args.dt, args.seed + i, stats, email), daemon=True)
        th.start()
        threads.append(th)
        time.sleep(0.5)
    try:
        while True:
            time.sleep(60)
            ok = sum(1 for v in stats.values() if "action" in v)
            print(time.strftime("%H:%M:%S"), f"{ok}/{len(stats)} agents ok;",
                  "mean score %.1f" % np.mean([v["score"] for v in stats.values() if "score" in v] or [0]), flush=True)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
