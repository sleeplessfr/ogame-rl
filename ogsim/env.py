"""The training interface: every player is an agent that sees numbers and picks one legal action per decision step.

    env = MultiEnv(Config(n_players=16))
    obs = env.reset()                                   # {player: float32 vector}
    obs, reward, done, info = env.step({0: (kind, a, b, c), 1: ...})   # players that give no action do nothing
    env.masks(0)                                        # which actions are legal right now (for masked policies)

Action = (kind, a, b, c), four small integers (see ACTION_NAMES / ARGS):
    0 noop
    1 build         a=planet slot  b=building
    2 research      b=technology
    3 build units   a=planet slot  b=unit (ship or defence)   c=share of the affordable amount (SHARES)
    4 espionage     a=planet slot  b=target slot              c=number of probes (PROBES)
    5 attack        a=planet slot  b=target slot              c=share of the war fleet (and cargo ships) sent (FLEET_SHARES)
    6 colonise      a=planet slot  b=position 1..15           c=system offset (OFFSETS)
    7 transport     a=from slot    b=to slot                  c=share of the stock (SHARES)
    8 recycle       a=planet slot  b=debris slot (nearest first)
Target slots are the other players sorted by distance from the acting player's home. An illegal action does nothing
(small penalty); it is the same refusal the real server would give.

Reward = change in score (points / 1000) since the last step, times `reward_scale`. Nothing else is rewarded: no rules, no tips.
"""
from __future__ import annotations

import math

import numpy as np

from .econdata import OBJECTS
from .unitdata import UNITS
from .world import ActionError, Config, NOT_ON_PLANETS, World, RESOURCES

MAXP = 6                  # planet slots
K_TARGETS = 8             # other players an agent can aim at
K_DEBRIS = 3
BUILDINGS = sorted(n for n, o in OBJECTS.items() if o["kind"] in ("building", "station") and n not in NOT_ON_PLANETS)
TECHS = sorted(n for n, o in OBJECTS.items() if o["kind"] == "research")
UNIT_NAMES = sorted(n for n in UNITS if "missile" not in n)
SHIPS = [n for n in UNIT_NAMES if UNITS[n]["kind"] == "ship"]
WAR_SHIPS = [n for n in SHIPS if UNITS[n]["attack"] > 5]
CARGO_SHIPS = [n for n in SHIPS if UNITS[n]["cargo"] >= 5000]
SHARES = (0.1, 0.25, 0.5, 1.0)
FLEET_SHARES = (0.25, 0.5, 0.75, 1.0)
PROBES = (1, 3, 5, 8)
OFFSETS = (-2, -1, 0, 1, 2)
ACTION_NAMES = ("noop", "build", "research", "units", "espionage", "attack", "colonise", "transport", "recycle")
ARGS = (len(ACTION_NAMES), MAXP, max(len(BUILDINGS), len(TECHS), len(UNIT_NAMES), K_TARGETS, 15), 5)   # sizes of kind, a, b, c
PLANET_FEATS = 4 + 4 + len(BUILDINGS) + 2 + len(UNIT_NAMES) + 3
OBS_SIZE = 6 + MAXP * PLANET_FEATS + len(TECHS) + 4 + 3 + K_TARGETS * 6 + K_DEBRIS * 3


def _lg(x: float) -> float:
    return math.log1p(max(0.0, x)) / 12.0


class MultiEnv:
    def __init__(self, cfg: Config | None = None, decision_seconds: float = 900.0, horizon_seconds: float = 7 * 86400.0,
                 reward_scale: float = 1.0, invalid_penalty: float = 0.001):
        self.cfg = cfg or Config()
        self.dt, self.horizon = decision_seconds, horizon_seconds
        self.reward_scale, self.invalid_penalty = reward_scale, invalid_penalty
        self.world: World | None = None
        self._last_score: list[float] = []
        self._targets: dict[int, list[int]] = {}

    @property
    def n(self) -> int:
        return self.cfg.n_players

    def reset(self, seed: int | None = None) -> dict[int, np.ndarray]:
        if seed is not None:
            self.cfg.seed = seed
        self.world = World(self.cfg)
        self._last_score = [0.0] * self.n
        self._targets = {}
        self._refresh_targets()
        return {i: self.observe(i) for i in range(self.n)}

    # ---- stepping ---------------------------------------------------------------------------------------------------
    def step(self, actions: dict[int, tuple]):
        w = self.world
        assert w is not None, "call reset() first"
        ok: dict[int, bool] = {}
        order = list(actions)
        w.rng.shuffle(order)
        for pid in order:
            ok[pid] = self.act(pid, actions[pid])
        w.advance(self.dt)
        self._refresh_targets()
        obs, rew, info = {}, {}, {"valid": ok}
        for i in range(self.n):
            sc = w.players[i].score
            rew[i] = (sc - self._last_score[i]) * self.reward_scale - (self.invalid_penalty if ok.get(i) is False else 0.0)
            self._last_score[i] = sc
            obs[i] = self.observe(i)
        return obs, rew, w.t >= self.horizon, info

    def _refresh_targets(self) -> None:
        w = self.world
        if self._targets:
            return   # home planets never move: the slots are fixed for the whole game
        homes = [pl.planets[0].coord for pl in w.players]
        for i in range(self.n):
            me = homes[i]
            others = [j for j in range(self.n) if j != i]
            others.sort(key=lambda j: (w.distance(me, homes[j]), j))
            self._targets[i] = others[:K_TARGETS]

    def act(self, pid: int, action) -> bool:
        """Apply one action. Returns False if the game would have refused it."""
        kind, a, b, c = (int(x) for x in action)
        w = self.world
        try:
            if kind == 0:
                return True
            pl = w.players[pid]
            if kind == 1:
                w.build(pid, a, BUILDINGS[b])
            elif kind == 2:
                w.research(pid, TECHS[b], 0)
            elif kind == 3:
                name = UNIT_NAMES[b]
                p = w._planet_of(pid, a)
                u = UNITS[name]
                fit = min((int(p.res[r] // u[r]) for r in RESOURCES if u[r] > 0), default=0)
                w.build_units(pid, a, name, max(1, int(fit * SHARES[c % 4])) if fit > 0 else 1)
            elif kind in (4, 5):
                tgt = self._target_planet(pid, b)
                p = w._planet_of(pid, a)
                if kind == 4:
                    n = min(PROBES[c % 4], p.ships.get("espionage_probe", 0))
                    w.send_fleet(pid, a, "espionage", tgt.coord, {"espionage_probe": n})
                else:
                    sh = FLEET_SHARES[c % 4]
                    ships = {n: int(p.ships.get(n, 0) * sh) for n in WAR_SHIPS + CARGO_SHIPS if int(p.ships.get(n, 0) * sh) > 0}
                    w.send_fleet(pid, a, "attack", tgt.coord, ships)
            elif kind == 6:
                p = w._planet_of(pid, a)
                s = (p.coord[1] - 1 + OFFSETS[c % 5]) % w.cfg.systems + 1
                w.send_fleet(pid, a, "colonisation", (p.coord[0], s, b % 15 + 1), {"colony_ship": 1})
            elif kind == 7:
                src = w._planet_of(pid, a)
                dst = w._planet_of(pid, b)
                if src is dst:
                    raise ActionError("same planet")
                ships = {n: src.ships.get(n, 0) for n in CARGO_SHIPS if src.ships.get(n, 0) > 0}
                cap = sum(UNITS[n]["cargo"] * k for n, k in ships.items())
                sh = SHARES[c % 4]
                want = {r: src.res[r] * sh for r in RESOURCES}
                tot = sum(want.values())
                if tot > cap > 0:
                    want = {r: v * cap / tot for r, v in want.items()}
                w.send_fleet(pid, a, "transport", dst.coord, ships, want)
            elif kind == 8:
                p = w._planet_of(pid, a)
                near = sorted(w.debris, key=lambda co: w.distance(p.coord, co))[:K_DEBRIS]
                if b >= len(near):
                    raise ActionError("no such debris field")
                w.send_fleet(pid, a, "recycle", near[b], {"recycler": p.ships.get("recycler", 0)})
            else:
                raise ActionError("unknown action")
            return True
        except (ActionError, IndexError, KeyError, ValueError):
            return False

    def _target_planet(self, pid: int, slot: int):
        ids = self._targets[pid]
        if slot >= len(ids):
            raise ActionError("no such target")
        return self.world.players[ids[slot]].planets[0]

    # ---- observation ------------------------------------------------------------------------------------------------
    def observe(self, pid: int) -> np.ndarray:
        w = self.world
        pl = w.players[pid]
        v = np.zeros(OBS_SIZE, dtype=np.float32)
        scores = sorted((q.score for q in w.players), reverse=True)
        rank = scores.index(pl.score) / max(1, self.n - 1)
        v[0:6] = (min(1.0, w.t / self.horizon), _lg(pl.score * 1000), rank, len(pl.planets) / MAXP, len(pl.fleets) / 8.0,
                  1.0 if pl.rq else 0.0)
        i = 6
        for slot in range(MAXP):
            if slot < len(pl.planets):
                p = pl.planets[slot]
                w.sync(p)
                f = [1.0, p.coord[2] / 15.0, p.used_fields() / max(1, p.max_fields()), _lg(p.max_fields() * 100)]
                f += [_lg(p.res[r]) for r in RESOURCES] + [min(1.0, p.res["metal"] / max(1.0, p.cap["metal"]))]
                f += [p.lv.get(b, 0) / 30.0 for b in BUILDINGS]
                f += [1.0 if p.bq else 0.0, 1.0 if p.uq else 0.0]
                f += [_lg(p.ships.get(u, 0) if UNITS[u]["kind"] == "ship" else p.defs.get(u, 0)) for u in UNIT_NAMES]
                f += [_lg(p.rate[r] * 3600) for r in RESOURCES]
                v[i:i + PLANET_FEATS] = f
            i += PLANET_FEATS
        v[i:i + len(TECHS)] = [pl.tech.get(t, 0) / 20.0 for t in TECHS]
        i += len(TECHS)
        inc = w.incoming(pid)
        v[i:i + 4] = (sum(1 for x in inc if x["mission"] == "attack") / 4.0, sum(1 for x in inc if x["mission"] == "espionage") / 4.0,
                      min([x["arrives_in"] for x in inc], default=86400.0) / 86400.0, len(pl.battles) / 50.0)
        i += 4
        home = pl.planets[0]
        v[i:i + 3] = (_lg(pl.looted), _lg(pl.ships_lost), len(w.debris) / 10.0)
        i += 3
        for slot in range(K_TARGETS):
            if slot < len(self._targets[pid]):
                j = self._targets[pid][slot]
                tp = w.players[j]
                rep = next((r for r in reversed(pl.intel) if r["owner"] == j), None)
                age = 1.0 if rep is None else min(1.0, (w.t - rep["t"]) / 86400.0)
                loot = _lg(sum(rep["resources"].values()) * 0.5) if rep else 0.0
                dpower = _lg(sum(UNITS[n]["hull"] * k / 10 + UNITS[n]["attack"] * k for n, k in rep["defense"].items())) if rep and "defense" in rep else -1.0
                v[i:i + 6] = (1.0, _lg(tp.score * 1000), min(1.0, w.distance(home.coord, tp.planets[0].coord) / 20000.0), age, loot, dpower)
            i += 6
        near = sorted(w.debris, key=lambda co: w.distance(home.coord, co))[:K_DEBRIS]
        for slot in range(K_DEBRIS):
            if slot < len(near):
                d = w.debris[near[slot]]
                v[i:i + 3] = (1.0, _lg(d["metal"] + d["crystal"]), min(1.0, w.distance(home.coord, near[slot]) / 20000.0))
            i += 3
        return v

    # ---- legal actions ----------------------------------------------------------------------------------------------
    def masks(self, pid: int) -> dict:
        """What is legal right now (True = allowed). Cheap; use it to keep a policy from wasting steps on refused actions."""
        w = self.world
        pl = w.players[pid]
        build = np.zeros((MAXP, len(BUILDINGS)), dtype=bool)
        units = np.zeros((MAXP, len(UNIT_NAMES)), dtype=bool)
        research = np.zeros(len(TECHS), dtype=bool)
        lab_planet = max(pl.planets, key=lambda q: q.lv.get("research_lab", 0))
        for slot, p in enumerate(pl.planets[:MAXP]):
            w.sync(p)
            if not p.bq:
                for k, name in enumerate(BUILDINGS):
                    cost = _price(name, p.lv.get(name, 0) + 1)
                    build[slot, k] = (w.requirements_met(pid, p, name) and all(p.res[r] >= cost[r] for r in RESOURCES)
                                      and (p.used_fields() < p.max_fields() or name == "terraformer")
                                      and not (name == "research_lab" and pl.rq) and not (name in ("shipyard", "nano_factory") and p.uq))
            if not (p.bq and p.bq[1] in ("shipyard", "nano_factory")):
                for k, name in enumerate(UNIT_NAMES):
                    u = UNITS[name]
                    units[slot, k] = w.requirements_met(pid, p, name) and all(p.res[r] >= u[r] for r in RESOURCES)
        if not pl.rq and lab_planet.lv.get("research_lab", 0) >= 1 and not any(q.bq and q.bq[1] == "research_lab" for q in pl.planets):
            home = pl.planets[0]
            for k, name in enumerate(TECHS):
                cost = _price(name, pl.tech.get(name, 0) + 1)
                research[k] = w.requirements_met(pid, lab_planet, name) and all(home.res[r] >= cost[r] for r in RESOURCES)
        slots_free = len(pl.fleets) < w.fleet_slots(pid)
        kind = np.zeros(len(ACTION_NAMES), dtype=bool)
        kind[0] = True
        kind[1] = build.any()
        kind[2] = research.any()
        kind[3] = units.any()
        have = lambda *names: any(p.ships.get(n, 0) > 0 for p in pl.planets for n in names)
        kind[4] = slots_free and have("espionage_probe")
        kind[5] = slots_free and have(*WAR_SHIPS)
        kind[6] = slots_free and have("colony_ship") and len(pl.planets) < w.max_planets(pid)
        kind[7] = slots_free and len(pl.planets) > 1 and have(*CARGO_SHIPS)
        kind[8] = slots_free and have("recycler") and bool(w.debris)
        return {"kind": kind, "build": build, "research": research, "units": units}


def _price(name: str, level: int) -> dict:
    from .econ import price
    return price(name, level)
