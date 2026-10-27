"""Simple players to train against and to measure against. They are opponents and yardsticks, not the thing being trained."""
from __future__ import annotations

import random

import numpy as np

from .env import BUILDINGS, TECHS, UNIT_NAMES, MultiEnv


class RandomBot:
    """Picks a uniformly random legal action (and does nothing about a third of the time)."""

    def __init__(self, seed: int = 0, idle: float = 0.3):
        self.rng = random.Random(seed)
        self.idle = idle

    def act(self, env: MultiEnv, pid: int) -> tuple:
        m = env.masks(pid)
        kinds = [k for k in range(1, len(m["kind"])) if m["kind"][k]]
        if not kinds or self.rng.random() < self.idle:
            return (0, 0, 0, 0)
        k = self.rng.choice(kinds)
        if k == 1:
            a, b = self.rng.choice(np.argwhere(m["build"]).tolist())
            return (1, a, b, 0)
        if k == 2:
            return (2, 0, self.rng.choice(np.flatnonzero(m["research"]).tolist()), 0)
        if k == 3:
            a, b = self.rng.choice(np.argwhere(m["units"]).tolist())
            return (3, a, b, self.rng.randrange(4))
        return (k, self.rng.randrange(len(env.world.players[pid].planets)), self.rng.randrange(8), self.rng.randrange(4))


ECO_ORDER = ["solar_plant", "metal_mine", "crystal_mine", "metal_mine", "solar_plant", "deuterium_synthesizer", "metal_mine", "crystal_mine",
             "robot_factory", "research_lab", "solar_plant", "metal_mine", "crystal_mine", "deuterium_synthesizer", "robot_factory",
             "metal_store", "crystal_store", "shipyard", "solar_plant", "metal_mine", "crystal_mine", "deuterium_synthesizer"]


class EcoBot:
    """A plain builder: mines and power in a fixed rhythm (keeping power ahead of use), then research and a modest defence. Never attacks."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, env: MultiEnv, pid: int) -> tuple:
        w = env.world
        pl = w.players[pid]
        p = pl.planets[0]
        m = env.masks(pid)
        want = None
        make_need = None
        from .econ import energy
        make, need = energy(p.lv, pl.tech.get("energy_technology", 0))
        if need > make * 0.9 and m["build"][0, BUILDINGS.index("solar_plant")]:
            return (1, 0, BUILDINGS.index("solar_plant"), 0)
        if m["kind"][2]:
            order = [t for t in ("energy_technology", "computer_technology", "espionage_technology", "combustion_drive", "armor_technology")]
            for t in order:
                if m["research"][TECHS.index(t)] and pl.tech.get(t, 0) < 6:
                    return (2, 0, TECHS.index(t), 0)
        if not p.bq:
            levels = {}
            for name in ECO_ORDER:
                levels[name] = levels.get(name, 0) + 1
                if p.lv.get(name, 0) < levels[name]:
                    if m["build"][0, BUILDINGS.index(name)]:
                        return (1, 0, BUILDINGS.index(name), 0)
                    break
            cands = np.argwhere(m["build"][0]).flatten().tolist()
            if cands and p.res["metal"] > 50000:
                return (1, 0, self.rng.choice(cands), 0)
        if m["units"][0, UNIT_NAMES.index("rocket_launcher")] and p.defs.get("rocket_launcher", 0) < 20 + w.t / 3600:
            return (3, 0, UNIT_NAMES.index("rocket_launcher"), 0)
        return (0, 0, 0, 0)
