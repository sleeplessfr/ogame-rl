#!/usr/bin/env python3
"""Diagnostic: does a second planet pay off once the bot is not limited to one action per long decision step?
A scripted builder that keeps every planet busy (power, storage, mines), with or without the colony chain, at several decision lengths."""
import sys
import numpy as np
sys.path.insert(0, ".")
from ogsim.bots import EcoBot
from ogsim.econ import energy
from ogsim.env import BUILDINGS, MultiEnv, TECHS, UNIT_NAMES
from ogsim.world import Config


class Spend:
    colonise = False

    def __init__(self, seed=0):
        self.k = 0

    def build(self, env, pid, slot, m):
        w = env.world; pl = w.players[pid]; p = pl.planets[slot]
        if p.bq or not m["kind"][1]:
            return None
        B = BUILDINGS.index
        ok = lambda n: m["build"][slot, B(n)]
        make, need = energy(p.lv, pl.tech.get("energy_technology", 0))
        if need > make * 0.95 and ok("solar_plant"):
            return (1, slot, B("solar_plant"), 0)
        for res, store in (("metal", "metal_store"), ("crystal", "crystal_store"), ("deuterium", "deuterium_store")):
            if p.res[res] > 0.85 * p.cap[res] and ok(store):
                return (1, slot, B(store), 0)
        if p.lv.get("robot_factory", 0) < 4 and ok("robot_factory"):
            return (1, slot, B("robot_factory"), 0)
        lv = lambda n: p.lv.get(n, 0)
        wants = sorted(["metal_mine", "crystal_mine", "deuterium_synthesizer"], key=lambda n: lv(n) * {"metal_mine": 1, "crystal_mine": 1.6, "deuterium_synthesizer": 2.5}[n])
        for n in wants:
            if ok(n):
                return (1, slot, B(n), 0)
        return None

    def act(self, env, pid):
        pl = env.world.players[pid]; m = env.masks(pid)
        a = self.special(env, pid, m)
        if a:
            return a
        n = len(pl.planets)
        for i in range(n):
            slot = (self.k + i) % n
            a = self.build(env, pid, slot, m)
            if a:
                self.k = slot + 1
                return a
        return (0, 0, 0, 0)

    def special(self, env, pid, m):
        pl = env.world.players[pid]; p = pl.planets[0]
        if m["kind"][2] and m["research"][TECHS.index("energy_technology")] and pl.tech.get("energy_technology", 0) < 5:
            return (2, 0, TECHS.index("energy_technology"), 0)
        if self.colonise and len(pl.planets) < 2:
            if not p.bq:
                for name, lv in (("research_lab", 3), ("shipyard", 4)):
                    if p.lv.get(name, 0) < lv and m["build"][0, BUILDINGS.index(name)]:
                        return (1, 0, BUILDINGS.index(name), 0)
            for t, lv in (("espionage_technology", 4), ("impulse_drive", 3), ("astrophysics", 2)):
                if pl.tech.get(t, 0) < lv:
                    if m["kind"][2] and m["research"][TECHS.index(t)]:
                        return (2, 0, TECHS.index(t), 0)
                    break
            if m["kind"][6]:
                return (6, 0, 6, 2)
            if pl.tech.get("astrophysics", 0) >= 2 and m["units"][0, UNIT_NAMES.index("colony_ship")] and not any("colony_ship" in f.ships for f in pl.fleets):
                return (3, 0, UNIT_NAMES.index("colony_ship"), 0)
        return None


class SpendCol(Spend):
    colonise = True


def game(cls, seed, days, dt):
    env = MultiEnv(Config(n_players=8, seed=seed, systems=60), decision_seconds=dt, horizon_seconds=days * 86400)
    env.reset(); bots = [cls(seed) if i == 0 else Spend(seed + i) for i in range(8)]
    done = False
    while not done:
        _, _, done, _ = env.step({i: bots[i].act(env, i) for i in range(8)})
    pl = env.world.players[0]
    return pl.score, len(pl.planets)


for dt in (3600, 600, 120):
    for days in (7, 14):
        for cls in (Spend, SpendCol):
            r = [game(cls, s, days, dt) for s in range(1, 5)]
            print(f"dt {dt:5d} days {days:2d} {cls.__name__:9s} score {np.mean([x[0] for x in r]):9.0f} planets {np.mean([x[1] for x in r]):.2f}", flush=True)
