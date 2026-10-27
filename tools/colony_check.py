#!/usr/bin/env python3
"""Diagnostic only (never used for training): does a second planet pay off in the simulated game's own score?
EcoBot vs EcoBot that also goes for a colony. Same seeds, same opponents (EcoBots)."""
import sys
import numpy as np
sys.path.insert(0, ".")
from ogsim.bots import EcoBot
from ogsim.env import BUILDINGS, MultiEnv, TECHS, UNIT_NAMES
from ogsim.world import Config


class ColBot(EcoBot):
    colonise = True

    def act(self, env, pid):
        w = env.world; pl = w.players[pid]; m = env.masks(pid)
        p = pl.planets[0]
        if len(pl.planets) < 2:
            if not p.bq:
                for name, lv in (("research_lab", 3), ("shipyard", 4), ("metal_store", 3), ("crystal_store", 3), ("deuterium_store", 2)):
                    if p.lv.get(name, 0) < lv and m["build"][0, BUILDINGS.index(name)]:
                        return (1, 0, BUILDINGS.index(name), 0)
            for t, lv in (("espionage_technology", 4), ("impulse_drive", 3), ("astrophysics", 2)):
                if pl.tech.get(t, 0) < lv:
                    if m["research"][TECHS.index(t)]:
                        return (2, 0, TECHS.index(t), 0)
            if self.colonise and m["kind"][6]:
                return (6, 0, 6, 2)
            if self.colonise and pl.tech.get("impulse_drive", 0) >= 3 and pl.tech.get("astrophysics", 0) >= 2 and m["units"][0, UNIT_NAMES.index("colony_ship")]:
                return (3, 0, UNIT_NAMES.index("colony_ship"), 0)
        return super().act(env, pid)


class NoColBot(ColBot):
    colonise = False


def game(cls, seed, days, dt):
    env = MultiEnv(Config(n_players=8, seed=seed, systems=60), decision_seconds=dt, horizon_seconds=days * 86400)
    env.reset(); bots = [cls(seed) if i == 0 else EcoBot(seed + i) for i in range(8)]
    done = False
    while not done:
        _, _, done, _ = env.step({i: bots[i].act(env, i) for i in range(8)})
    return env.world.players[0].score, len(env.world.players[0].planets)


for days in (14, 30):
    for cls in (NoColBot, ColBot):
        r = [game(cls, s, days, 3600) for s in range(1, 7)]
        print(days, cls.__name__, "score %.0f" % np.mean([x[0] for x in r]), "planets %.2f" % np.mean([x[1] for x in r]))
