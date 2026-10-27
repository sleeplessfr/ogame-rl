"""One battle and its aftermath, using the game's rules (6 rounds, rapid fire, shields that come back each round, hull explosion)
and the game's loot / debris / rebuild rules (LootService, BattleEngine::calculateDebris, defence repair).

Units are the names of ogsim.unitdata.UNITS. Fleets are plain dicts {name: count}. Technology = (weapons, shielding, armour) levels.
Big fleets are scaled down (both sides equally) to keep a fight cheap: the result is scaled back up.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .unitdata import UNITS

ROUNDS = 6
MAX_UNITS = 2500         # a fight with more units than this is simulated with fewer, equal shares
HULL_DIVISOR = 10        # the battle engine uses a tenth of the listed structural integrity
DEFENCE_REPAIR = 0.7     # share of destroyed defence units that come back
LOOT_SHARE = 0.5         # share of a planet's resources that can be taken
DEBRIS_FROM_SHIPS = 0.30
DEBRIS_FROM_DEFENCE = 0.0
Tech = tuple[int, int, int]


def unit_cost(name: str) -> int:
    u = UNITS[name]
    return u["metal"] + u["crystal"] + u["deuterium"]


def _expand(fleet: dict[str, int], tech: Tech, scale: float) -> list[list]:
    """One entry per unit: [name, attack, shield_max, hull_max, hull, shield]."""
    w, sh, ar = tech
    out = []
    for name, n in fleet.items():
        u = UNITS.get(name)
        if not u or n <= 0:
            continue
        attack, shield, hull = u["attack"] * (1 + 0.1 * w), u["shield"] * (1 + 0.1 * sh), u["hull"] / HULL_DIVISOR * (1 + 0.1 * ar)
        for _ in range(max(1, int(round(n * scale)))):
            out.append([name, attack, shield, hull, hull, shield])
    return out


def _fight(att: list[list], dfn: list[list], rng: random.Random) -> int:
    """Plays up to 6 rounds in place; returns the number of rounds fought."""
    rounds = 0
    for _ in range(ROUNDS):
        if not att or not dfn:
            break
        rounds += 1
        for u in att + dfn:
            u[5] = u[2]
        for shooters, targets in ((att, dfn), (dfn, att)):
            pool = list(targets)
            for shooter in list(shooters):
                if not pool:
                    break
                rapid = UNITS[shooter[0]]["rapid"]
                while True:
                    target = pool[rng.randrange(len(pool))]
                    dmg = shooter[1]
                    if dmg < 0.01 * target[2]:
                        break
                    absorbed = min(target[5], dmg)
                    target[5] -= absorbed
                    target[4] -= dmg - absorbed
                    if 0 < target[4] < 0.7 * target[3] and rng.random() < 1 - target[4] / target[3]:
                        target[4] = 0
                    rf = rapid.get(target[0], 0)
                    if rf <= 1 or rng.random() >= 1 - 1 / rf:
                        break
        att[:] = [u for u in att if u[4] > 0]
        dfn[:] = [u for u in dfn if u[4] > 0]
    return rounds


def split_loot(loot: dict[str, float], capacity: int) -> dict[str, float]:
    """The game's even sharing of the cargo hold between metal, crystal and deuterium."""
    keys = ("metal", "crystal", "deuterium")
    if capacity >= sum(loot[k] for k in keys):
        return {k: float(loot[k]) for k in keys}
    out = {k: min(loot[k], math.floor(capacity / 3)) for k in keys}
    for _ in range(3):
        room = capacity - sum(out.values())
        open_ = [k for k in keys if loot[k] > out[k]]
        if room <= 0 or not open_:
            break
        for k in open_:
            out[k] = min(loot[k], out[k] + room / len(open_))
    return out


@dataclass
class Outcome:
    attacker_won: bool
    rounds: int
    attacker_left: dict[str, int]
    defender_left: dict[str, int]      # ships and defence after the fight (defence already rebuilt)
    attacker_lost: dict[str, int]
    defender_lost: dict[str, int]      # permanently lost (defence: after the rebuild)
    loot: dict[str, float]
    debris: dict[str, float]


def fight(attackers: dict[str, int], defenders: dict[str, int], resources: dict[str, float], rng: random.Random,
          attacker_tech: Tech = (0, 0, 0), defender_tech: Tech = (0, 0, 0), capacity: int | None = None) -> Outcome:
    """`defenders` = ships and defence together; `resources` = what the target planet holds; `capacity` = cargo hold of the
    attacking fleet that is left after the fight (None = computed from the survivors)."""
    total = sum(attackers.values()) + sum(defenders.values())
    scale = min(1.0, MAX_UNITS / total) if total else 1.0
    att, dfn = _expand(attackers, attacker_tech, scale), _expand(defenders, defender_tech, scale)
    att0, dfn0 = list(att), list(dfn)
    rounds = _fight(att, dfn, rng) if att and dfn else 0
    won = bool(att) and not dfn
    a_alive: dict[str, int] = {}
    for u in att:
        a_alive[u[0]] = a_alive.get(u[0], 0) + 1
    d_alive: dict[str, int] = {}
    for u in dfn:
        d_alive[u[0]] = d_alive.get(u[0], 0) + 1
    a_left = {n: min(k, int(round(a_alive.get(n, 0) / scale))) for n, k in attackers.items() if k > 0}
    d_left, d_lost = {}, {}
    for n, k in defenders.items():
        if k <= 0:
            continue
        alive = min(k, int(round(d_alive.get(n, 0) / scale)))
        dead = k - alive
        if UNITS[n]["kind"] == "defense":
            back = int(round(dead * DEFENCE_REPAIR))
            alive, dead = alive + back, dead - back
        d_left[n], d_lost[n] = alive, dead
    a_lost = {n: k - a_left.get(n, 0) for n, k in attackers.items() if k > 0}
    debris = {"metal": 0.0, "crystal": 0.0, "deuterium": 0.0}
    for n, k in a_lost.items():
        u = UNITS[n]
        debris["metal"] += u["metal"] * k * DEBRIS_FROM_SHIPS
        debris["crystal"] += u["crystal"] * k * DEBRIS_FROM_SHIPS
    for n, k in d_lost.items():
        u = UNITS[n]
        share = DEBRIS_FROM_DEFENCE if u["kind"] == "defense" else DEBRIS_FROM_SHIPS
        debris["metal"] += u["metal"] * k * share
        debris["crystal"] += u["crystal"] * k * share
    loot = {"metal": 0.0, "crystal": 0.0, "deuterium": 0.0}
    if won:
        cap = capacity if capacity is not None else sum(UNITS[n]["cargo"] * k for n, k in a_left.items())
        want = {r: max(0.0, resources.get(r, 0.0)) * LOOT_SHARE for r in loot}
        loot = split_loot(want, cap)
    return Outcome(won, rounds, a_left, d_left, a_lost, d_lost, loot, debris)
