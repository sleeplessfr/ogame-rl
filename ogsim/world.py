"""The fast universe. Every player acts only through the game's legal actions (the same ones the real server's API offers):

    build(pid, planet, name)                    queue a building upgrade on a planet
    research(pid, name)                         queue a research
    build_units(pid, planet, name, amount)      ships and defence
    send_fleet(pid, planet, mission, target, ships, cargo, speed_pct)   espionage / attack / transport / colonisation / recycle
    state(pid), highscore(), reports(pid), incoming(pid)               what the player may look at
    advance(seconds)                            time passes; everything happens in order

Time is event driven (building and research finish, fleets arrive), so a day of game time costs only the actions taken.
Production is integrated exactly between events. Formulas come from the game source via econdata.py / unitdata.py (generated).
Left out on purpose: moons, missiles, expeditions, ACS, officers, items, merchant, jump gate, vacation mode.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

from . import battle, econ
from .econdata import OBJECTS
from .unitdata import UNITS

RESOURCES = ("metal", "crystal", "deuterium")
ENGINE_BONUS = {"combustion_drive": 0.1, "impulse_drive": 0.2, "hyperspace_drive": 0.3}
HOME_FIELDS = 163
START_RES = (500.0, 500.0, 0.0)
MISSIONS = ("espionage", "attack", "transport", "colonisation", "recycle")
NOT_ON_PLANETS = {"lunar_base", "sensor_phalanx", "jump_gate", "missile_silo"}


@dataclass
class Config:
    n_players: int = 20
    seed: int = 1
    economy_speed: float = 100.0
    fleet_speed: float = 5.0          # divides all flight times (the owner's server: war speed)
    galaxies: int = 1
    systems: int = 499
    start_res: tuple = START_RES
    protection_points: float = 0.0    # players under this score are safe from players with 5x their score (0 = no protection)
    max_planets_base: int = 1         # planets allowed = this + astrophysics level (rounded up half)


def fields_for(position: int, rng: random.Random) -> int:
    mean = {1: 70, 2: 80, 3: 100, 4: 150, 5: 160, 6: 170, 7: 190, 8: 200, 9: 190, 10: 170, 11: 150, 12: 130, 13: 100, 14: 80, 15: 70}[position]
    return max(40, int(rng.gauss(mean, mean * 0.12)))


def temperature(position: int) -> float:
    """Average temperature of a planet position (the API does not give it; midpoint of the game's range minus 20)."""
    lo = {1: 220, 2: 170, 3: 120, 4: 70, 5: 60, 6: 50, 7: 40, 8: 30, 9: 20, 10: 10, 11: 0, 12: -10, 13: -50, 14: -90, 15: -130}[position]
    return lo - 20.0


class Planet:
    __slots__ = ("id", "owner", "coord", "base_fields", "temp", "res", "lv", "bq", "uq", "ships", "defs", "t", "rate", "cap", "uq_t")

    def __init__(self, pid: int, owner: int, coord: tuple, base_fields: int, res: tuple):
        self.id, self.owner, self.coord = pid, owner, coord
        self.base_fields, self.temp = base_fields, temperature(coord[2])
        self.res = {r: float(v) for r, v in zip(RESOURCES, res)}
        self.lv: dict[str, int] = {}
        self.bq: tuple | None = None          # (finish, name, level)
        self.uq: list[list] = []              # [name, remaining, seconds_each, next_finish]
        self.ships: dict[str, int] = {}
        self.defs: dict[str, int] = {}
        self.t = 0.0
        self.rate: dict[str, float] = {r: 0.0 for r in RESOURCES}
        self.cap: dict[str, int] = econ.storage({})
        self.uq_t = 0.0

    def used_fields(self) -> int:
        return sum(self.lv.values())

    def max_fields(self) -> int:
        return econ.fields_max(self.base_fields, self.lv.get("terraformer", 0))


class Player:
    def __init__(self, pid: int, name: str):
        self.id, self.name = pid, name
        self.planets: list[Planet] = []
        self.tech: dict[str, int] = {}
        self.rq: tuple | None = None          # (finish, name, level, planet)
        self.points = 0.0                     # resource value ever put into buildings, research and units (still alive), as the game scores
        self.fleets: list["Fleet"] = []
        self.intel: list[dict] = []           # espionage reports, newest last
        self.battles: list[dict] = []
        self.ships_lost = 0.0
        self.looted = 0.0

    @property
    def score(self) -> float:
        return self.points / 1000.0


@dataclass
class Fleet:
    id: int
    owner: int
    origin: int                    # planet id
    target: tuple
    mission: str
    ships: dict
    cargo: dict
    t_depart: float
    t_arrive: float
    t_return: float
    phase: str = "out"
    target_owner: int | None = None
    speed_pct: float = 100.0


class World:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.rng = random.Random(self.cfg.seed)
        self.t = 0.0
        self.players: list[Player] = []
        self.planets: dict[int, Planet] = {}
        self.by_coord: dict[tuple, Planet] = {}
        self.debris: dict[tuple, dict] = {}
        self.events: list = []
        self._seq = 0
        self._fleet_id = 0
        self._planet_id = 0
        for i in range(self.cfg.n_players):
            self._new_player(i)

    # ---- set up -----------------------------------------------------------------------------------------------------
    def _free_coord(self, lo: int = 4, hi: int = 12) -> tuple:
        while True:
            c = (self.rng.randint(1, self.cfg.galaxies), self.rng.randint(1, self.cfg.systems), self.rng.randint(lo, hi))
            if c not in self.by_coord:
                return c

    def _new_planet(self, owner: int, coord: tuple, fields: int, res: tuple = (0.0, 0.0, 0.0)) -> Planet:
        self._planet_id += 1
        p = Planet(self._planet_id, owner, coord, fields, res)
        p.t = self.t
        p.uq_t = self.t
        self.planets[p.id] = p
        self.by_coord[coord] = p
        self.players[owner].planets.append(p)
        self._reprice(p)
        return p

    def _new_player(self, pid: int) -> Player:
        pl = Player(pid, f"player{pid}")
        self.players.append(pl)
        self._new_planet(pid, self._free_coord(), HOME_FIELDS, self.cfg.start_res)
        return pl

    # ---- production -----------------------------------------------------------------------------------------------
    def _reprice(self, p: Planet) -> None:
        pl = self.players[p.owner]
        p.rate = {r: v / 3600.0 for r, v in econ.production(p.lv, p.coord[2], p.temp, self.cfg.economy_speed,
                                                            pl.tech.get("energy_technology", 0), pl.tech.get("plasma_technology", 0)).items()}
        p.cap = econ.storage(p.lv)

    def sync(self, p: Planet, t: float | None = None) -> None:
        """Bring a planet up to time t: stock grows at the current rate up to the storage cap, finished units appear."""
        t = self.t if t is None else t
        dt = t - p.t
        if dt > 0:
            for r in RESOURCES:
                v = p.res[r] + p.rate[r] * dt
                if p.rate[r] > 0 and p.res[r] < p.cap[r]:
                    v = min(v, float(p.cap[r]))
                elif p.rate[r] > 0:
                    v = p.res[r]
                p.res[r] = max(0.0, v)
            p.t = t
        while p.uq and p.uq[0][3] <= t:
            q = p.uq[0]
            done = min(q[1], 1 + int((t - q[3]) // q[2]))
            bucket = p.defs if UNITS[q[0]]["kind"] == "defense" else p.ships
            bucket[q[0]] = bucket.get(q[0], 0) + done
            q[1] -= done
            if q[1] <= 0:
                p.uq.pop(0)
            else:
                q[3] += done * q[2]
                break

    # ---- events ----------------------------------------------------------------------------------------------------
    def _push(self, when: float, kind: str, payload) -> None:
        self._seq += 1
        heapq.heappush(self.events, (when, self._seq, kind, payload))

    def advance(self, seconds: float) -> None:
        end = self.t + seconds
        while self.events and self.events[0][0] <= end:
            when, _, kind, payload = heapq.heappop(self.events)
            self.t = max(self.t, when)
            getattr(self, "_on_" + kind)(payload)
        self.t = end
        for p in self.planets.values():
            self.sync(p)

    # ---- legal actions -----------------------------------------------------------------------------------------------
    def _planet_of(self, pid: int, planet: int) -> Planet:
        pl = self.players[pid]
        if not 0 <= planet < len(pl.planets):
            raise ActionError("no such planet")
        p = pl.planets[planet]
        self.sync(p)
        return p

    def requirements_met(self, pid: int, p: Planet, name: str) -> bool:
        pl = self.players[pid]
        need = (OBJECTS.get(name) or UNITS.get(name) or {}).get("requires", {})
        for n, lvl in need.items():
            have = pl.tech.get(n, 0) if OBJECTS.get(n, {}).get("kind") == "research" else p.lv.get(n, 0)
            if have < lvl:
                return False
        return True

    @staticmethod
    def _pay(p: Planet, cost: dict) -> None:
        for r in RESOURCES:
            p.res[r] -= cost.get(r, 0)

    @staticmethod
    def _affordable(p: Planet, cost: dict) -> bool:
        return all(p.res[r] >= cost.get(r, 0) for r in RESOURCES)

    def build(self, pid: int, planet: int, name: str) -> float:
        """Queue the next level of a building. Returns the seconds it takes."""
        p = self._planet_of(pid, planet)
        o = OBJECTS.get(name)
        if not o or o["kind"] not in ("building", "station") or name in NOT_ON_PLANETS:
            raise ActionError(f"{name} is not a building")
        pl = self.players[pid]
        if p.bq:
            raise ActionError("the building queue is busy")
        if name == "research_lab" and pl.rq:
            raise ActionError("the lab is busy researching")
        if name in ("shipyard", "nano_factory") and p.uq:
            raise ActionError("the shipyard is busy building units")
        if not self.requirements_met(pid, p, name):
            raise ActionError("requirements not met")
        if p.used_fields() >= p.max_fields() and name != "terraformer":
            raise ActionError("no free fields")
        level = p.lv.get(name, 0) + 1
        cost = econ.price(name, level)
        if not self._affordable(p, cost):
            raise ActionError("not enough resources")
        self._pay(p, cost)
        secs = econ.build_seconds(name, level, p.lv.get("robot_factory", 0), p.lv.get("nano_factory", 0), self.cfg.economy_speed)
        p.bq = (self.t + secs, name, level)
        pl.points += sum(cost.values())
        self._push(p.bq[0], "building", (p.id, name, level))
        return float(secs)

    def _on_building(self, payload) -> None:
        pid, name, level = payload
        p = self.planets[pid]
        self.sync(p)
        p.lv[name] = level
        p.bq = None
        self._reprice(p)

    def research(self, pid: int, name: str, planet: int = 0) -> float:
        pl = self.players[pid]
        o = OBJECTS.get(name)
        if not o or o["kind"] != "research":
            raise ActionError(f"{name} is not a research")
        p = self._planet_of(pid, planet)
        if pl.rq:
            raise ActionError("a research is already running")
        lab = max(q.lv.get("research_lab", 0) for q in pl.planets)
        if lab < 1 or not self.requirements_met(pid, p if p.lv.get("research_lab") else max(pl.planets, key=lambda q: q.lv.get("research_lab", 0)), name):
            raise ActionError("requirements not met")
        if any(q.bq and q.bq[1] == "research_lab" for q in pl.planets):
            raise ActionError("the lab is being upgraded")
        level = pl.tech.get(name, 0) + 1
        cost = econ.price(name, level)
        if not self._affordable(p, cost):
            raise ActionError("not enough resources")
        self._pay(p, cost)
        secs = econ.research_seconds(name, level, lab, self.cfg.economy_speed)
        pl.rq = (self.t + secs, name, level)
        pl.points += sum(cost.values())
        self._push(pl.rq[0], "research", (pid, name, level))
        return float(secs)

    def _on_research(self, payload) -> None:
        pid, name, level = payload
        pl = self.players[pid]
        for p in pl.planets:
            self.sync(p)
        pl.tech[name] = level
        pl.rq = None
        for p in pl.planets:
            self._reprice(p)

    def build_units(self, pid: int, planet: int, name: str, amount: int) -> float:
        u = UNITS.get(name)
        if not u or "missile" in name:
            raise ActionError(f"{name} cannot be built")
        p = self._planet_of(pid, planet)
        if amount < 1:
            raise ActionError("amount must be positive")
        if not self.requirements_met(pid, p, name):
            raise ActionError("requirements not met")
        if p.bq and p.bq[1] in ("shipyard", "nano_factory"):
            raise ActionError("the shipyard is being upgraded")
        if name in ("small_shield_dome", "large_shield_dome"):
            have = p.defs.get(name, 0) + sum(q[1] for q in p.uq if q[0] == name)
            if have + amount > 1:
                raise ActionError("only one shield dome per planet")
            amount = 1
        cost = {r: u[r] * amount for r in RESOURCES}
        if not self._affordable(p, cost):
            raise ActionError("not enough resources")
        self._pay(p, cost)
        each = max(1.0, math.floor(u["hull"] / (2500 * (1 + p.lv.get("shipyard", 0)) * self.cfg.economy_speed * 2 ** p.lv.get("nano_factory", 0)) * 3600))
        start = self.t
        if p.uq:
            last = p.uq[-1]
            start = last[3] + (last[1] - 1) * last[2]   # the last queued unit finishes here
        p.uq.append([name, amount, each, start + each])
        self.players[pid].points += sum(cost.values())
        return each * amount

    # ---- fleets ----------------------------------------------------------------------------------------------------
    def ship_speed(self, pid: int, name: str) -> float:
        u, tech = UNITS[name], self.players[pid].tech
        base = u["speed"]
        engine = next((d for d in ENGINE_BONUS if d in u["requires"]), "combustion_drive")
        for drive, lvl, newspeed in u["upgrades"]:
            if tech.get(drive, 0) >= lvl:
                engine = drive
                if newspeed:
                    base = newspeed
        return base * (1 + ENGINE_BONUS[engine] * tech.get(engine, 0))

    def distance(self, a: tuple, b: tuple) -> int:
        dg, ds, dp = abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])
        if dg:
            return min(dg, abs(dg - self.cfg.galaxies)) * 20000
        if ds:
            return max(min(ds, abs(ds - self.cfg.systems)), 1) * 95 + 2700
        if dp:
            return dp * 5 + 1000
        return 5

    def flight_seconds(self, pid: int, a: tuple, b: tuple, ships: dict, speed_pct: float = 100.0) -> int:
        slowest = min(self.ship_speed(pid, n) for n, k in ships.items() if k > 0)
        return max(1, round((35000 / (speed_pct / 10.0) * math.sqrt(self.distance(a, b) * 10 / slowest) + 10) / self.cfg.fleet_speed))

    def fuel_cost(self, pid: int, a: tuple, b: tuple, ships: dict, speed_pct: float = 100.0) -> int:
        dist = self.distance(a, b)
        dur = self.flight_seconds(pid, a, b, ships, speed_pct)
        speed_value = max(0.5, dur * self.cfg.fleet_speed - 10)
        total = 0.0
        for n, k in ships.items():
            if k <= 0:
                continue
            s = 35000 / speed_value * math.sqrt(dist * 10 / self.ship_speed(pid, n))
            total += max(UNITS[n]["fuel"] * k * dist / 35000 * (s / 10 + 1) ** 2, 1)
        return int(round(total))

    def fleet_slots(self, pid: int) -> int:
        return 1 + self.players[pid].tech.get("computer_technology", 0)

    def max_planets(self, pid: int) -> int:
        return self.cfg.max_planets_base + (self.players[pid].tech.get("astrophysics", 0) + 1) // 2

    def _protected(self, att: Player, tgt: Player) -> bool:
        lim = self.cfg.protection_points
        if lim <= 0:
            return False
        a, d = att.score, tgt.score
        return (d < lim and a > 5 * max(d, 1.0)) or (a < lim and d > 5 * max(a, 1.0))

    def send_fleet(self, pid: int, planet: int, mission: str, target: tuple, ships: dict, cargo: dict | None = None,
                   speed_pct: float = 100.0) -> int:
        """Send ships on a mission. Returns the fleet id."""
        if mission not in MISSIONS:
            raise ActionError(f"unknown mission {mission}")
        pl = self.players[pid]
        p = self._planet_of(pid, planet)
        ships = {n: int(k) for n, k in ships.items() if int(k) > 0}
        cargo = {r: float(max(0, (cargo or {}).get(r, 0))) for r in RESOURCES}
        if not ships:
            raise ActionError("no ships")
        for n, k in ships.items():
            if n not in UNITS or UNITS[n]["kind"] != "ship" or p.ships.get(n, 0) < k:
                raise ActionError(f"not enough {n}")
        if len([f for f in pl.fleets]) >= self.fleet_slots(pid):
            raise ActionError("no free fleet slot")
        target = tuple(target)
        if not (1 <= target[0] <= self.cfg.galaxies and 1 <= target[1] <= self.cfg.systems and 1 <= target[2] <= 15):
            raise ActionError("no such coordinates")
        dest = self.by_coord.get(target)
        t_owner = dest.owner if dest else None
        if mission in ("espionage", "attack"):
            if dest is None or dest.owner == pid:
                raise ActionError("needs a foreign planet")
            if self._protected(pl, self.players[dest.owner]):
                raise ActionError("the target is under protection")
            if mission == "espionage" and not ships.get("espionage_probe"):
                raise ActionError("espionage needs probes")
            if mission == "attack" and not any(UNITS[n]["attack"] > 0 for n in ships):
                raise ActionError("nothing that can fight")
        if mission == "transport" and dest is None:
            raise ActionError("no planet there")
        if mission == "colonisation":
            if not ships.get("colony_ship"):
                raise ActionError("needs a colony ship")
            if dest is not None:
                raise ActionError("the position is taken")
            if len(pl.planets) >= self.max_planets(pid):
                raise ActionError("planet limit reached")
        if mission == "recycle" and not ships.get("recycler"):
            raise ActionError("needs recyclers")
        fuel = self.fuel_cost(pid, p.coord, target, ships, speed_pct)
        if p.res["deuterium"] < fuel + cargo["deuterium"]:
            raise ActionError("not enough deuterium for the flight")
        capacity = sum(UNITS[n]["cargo"] * k for n, k in ships.items())
        if sum(cargo.values()) > capacity:
            raise ActionError("cargo does not fit")
        for r in RESOURCES:
            if p.res[r] < cargo[r] + (fuel if r == "deuterium" else 0):
                raise ActionError("not enough resources to load")
        for r in RESOURCES:
            p.res[r] -= cargo[r] + (fuel if r == "deuterium" else 0)
        for n, k in ships.items():
            p.ships[n] -= k
        dur = self.flight_seconds(pid, p.coord, target, ships, speed_pct)
        self._fleet_id += 1
        f = Fleet(self._fleet_id, pid, p.id, target, mission, ships, cargo, self.t, self.t + dur, self.t + 2 * dur,
                  target_owner=t_owner, speed_pct=speed_pct)
        pl.fleets.append(f)
        self._push(f.t_arrive, "arrive", f)
        return f.id

    def _on_arrive(self, f: Fleet) -> None:
        getattr(self, "_arrive_" + f.mission)(f)

    def _go_home(self, f: Fleet) -> None:
        f.phase = "back"
        f.t_return = self.t + (f.t_arrive - f.t_depart)
        if sum(f.ships.values()) <= 0 and sum(f.cargo.values()) <= 0:
            self._remove(f)
            return
        self._push(f.t_return, "home", f)

    def _remove(self, f: Fleet) -> None:
        fl = self.players[f.owner].fleets
        if f in fl:
            fl.remove(f)

    def _on_home(self, f: Fleet) -> None:
        origin = self.planets.get(f.origin)
        if origin is not None and origin.owner == f.owner:
            self.sync(origin)
            for n, k in f.ships.items():
                origin.ships[n] = origin.ships.get(n, 0) + k
            for r in RESOURCES:
                origin.res[r] += f.cargo.get(r, 0.0)
        self._remove(f)

    def _arrive_transport(self, f: Fleet) -> None:
        dest = self.by_coord.get(f.target)
        if dest is not None:
            self.sync(dest)
            for r in RESOURCES:
                dest.res[r] += f.cargo.get(r, 0.0)
            f.cargo = {r: 0.0 for r in RESOURCES}
        self._go_home(f)

    def _arrive_colonisation(self, f: Fleet) -> None:
        pl = self.players[f.owner]
        if f.target not in self.by_coord and len(pl.planets) < self.max_planets(f.owner):
            self._new_planet(f.owner, f.target, fields_for(f.target[2], self.rng), (500.0, 500.0, 0.0))
            f.ships["colony_ship"] -= 1
            f.ships = {n: k for n, k in f.ships.items() if k > 0}
        self._go_home(f)

    def _arrive_recycle(self, f: Fleet) -> None:
        field_ = self.debris.get(f.target)
        if field_:
            cap = sum(UNITS[n]["cargo"] * k for n, k in f.ships.items() if n == "recycler") - sum(f.cargo.values())
            total = field_["metal"] + field_["crystal"]
            if total > 0 and cap > 0:
                take = min(1.0, cap / total)
                for r in ("metal", "crystal"):
                    got = field_[r] * take
                    field_[r] -= got
                    f.cargo[r] += got
                if field_["metal"] + field_["crystal"] < 1:
                    self.debris.pop(f.target, None)
        self._go_home(f)

    def _arrive_espionage(self, f: Fleet) -> None:
        dest = self.by_coord.get(f.target)
        att = self.players[f.owner]
        if dest is None:
            self._go_home(f)
            return
        dfn = self.players[dest.owner]
        self.sync(dest)
        probes = f.ships.get("espionage_probe", 0)
        a_lvl, d_lvl = att.tech.get("espionage_technology", 0), dfn.tech.get("espionage_technology", 0)
        ship_count = sum(k for k in dest.ships.values())
        factor = d_lvl - a_lvl + 1
        chance = 0
        if probes > 0 and ship_count > 0 and factor > 0:
            chance = int(min(100, max(0, math.floor(ship_count * factor / (probes * 4) * 100))))
        report: dict = {"t": self.t, "target": f.target, "owner": dest.owner, "resources": {r: int(dest.res[r]) for r in RESOURCES}}
        if chance and self.rng.random() * 100 < chance:
            out = battle.fight(f.ships, dict(dest.ships), {}, self.rng, (0, 0, 0), self._tech(dfn))
            for n, k in out.attacker_lost.items():
                f.ships[n] -= k
            self._apply_defender(dest, out, ships_only=True)
            self._add_debris(f.target, out.debris)
            f.ships = {n: k for n, k in f.ships.items() if k > 0}
            report["countered"] = True
            dfn.battles.append({"t": self.t, "kind": "counter_espionage", "at": f.target})
        probes = f.ships.get("espionage_probe", 0)
        remaining = max(0, probes - max(0, d_lvl - a_lvl) ** 2)
        if probes > 0:
            def reveal(need_probes: int, need_lvl: int) -> bool:
                return remaining >= need_probes or a_lvl - need_lvl >= d_lvl
            if reveal(2, 1):
                report["ships"] = dict(dest.ships)
            if reveal(3, 2):
                report["defense"] = dict(dest.defs)
            if reveal(5, 3):
                report["buildings"] = dict(dest.lv)
            if reveal(7, 4):
                report["research"] = dict(dfn.tech)
            att.intel.append(report)
            self._go_home(f)
        else:
            self._remove(f)
            att.battles.append({"t": self.t, "kind": "lost_contact", "at": f.target})

    def _tech(self, pl: Player) -> tuple:
        return (pl.tech.get("weapon_technology", 0), pl.tech.get("shielding_technology", 0), pl.tech.get("armor_technology", 0))

    def _apply_defender(self, dest: Planet, out: battle.Outcome, ships_only: bool = False) -> None:
        dfn = self.players[dest.owner]
        for n, k in out.defender_lost.items():
            bucket = dest.defs if UNITS[n]["kind"] == "defense" else dest.ships
            bucket[n] = max(0, bucket.get(n, 0) - k)
            dfn.points -= k * battle.unit_cost(n)
            dfn.ships_lost += k * battle.unit_cost(n)

    def _add_debris(self, coord: tuple, debris: dict) -> None:
        if debris["metal"] + debris["crystal"] < 1:
            return
        d = self.debris.setdefault(coord, {"metal": 0.0, "crystal": 0.0})
        d["metal"] += debris["metal"]
        d["crystal"] += debris["crystal"]

    def _arrive_attack(self, f: Fleet) -> None:
        dest = self.by_coord.get(f.target)
        att = self.players[f.owner]
        if dest is None or dest.owner == f.owner:
            self._go_home(f)
            return
        dfn = self.players[dest.owner]
        self.sync(dest)
        defenders = {**{n: k for n, k in dest.ships.items() if k > 0}, **{n: k for n, k in dest.defs.items() if k > 0}}
        cap = sum(UNITS[n]["cargo"] * k for n, k in f.ships.items()) - sum(f.cargo.values())
        out = battle.fight(f.ships, defenders, dest.res, self.rng, self._tech(att), self._tech(dfn), capacity=None)
        left_cap = sum(UNITS[n]["cargo"] * k for n, k in out.attacker_left.items()) - sum(f.cargo.values())
        loot = battle.split_loot({r: out.loot[r] for r in RESOURCES}, max(0, int(left_cap))) if out.attacker_won else out.loot
        for r in RESOURCES:
            dest.res[r] = max(0.0, dest.res[r] - loot[r])
            f.cargo[r] += loot[r]
        lost_value = 0.0
        for n, k in out.attacker_lost.items():
            lost_value += k * battle.unit_cost(n)
        att.points -= lost_value
        att.ships_lost += lost_value
        att.looted += sum(loot.values())
        f.ships = {n: k for n, k in out.attacker_left.items() if k > 0}
        self._apply_defender(dest, out)
        self._add_debris(f.target, out.debris)
        rec = {"t": self.t, "kind": "battle", "at": f.target, "attacker": f.owner, "defender": dest.owner, "won": out.attacker_won,
               "rounds": out.rounds, "loot": loot, "attacker_lost": out.attacker_lost, "defender_lost": out.defender_lost,
               "attacker_value_lost": lost_value}
        att.battles.append(rec)
        dfn.battles.append(rec)
        self._go_home(f)

    # ---- what a player may look at -----------------------------------------------------------------------------------
    def highscore(self) -> list[dict]:
        rows = sorted(({"player": pl.id, "score": round(pl.score, 1), "home": pl.planets[0].coord, "planets": len(pl.planets)}
                       for pl in self.players), key=lambda r: -r["score"])
        for i, r in enumerate(rows):
            r["rank"] = i + 1
        return rows

    def incoming(self, pid: int) -> list[dict]:
        """Foreign fleets on their way to this player's planets (the real /fleet shows the same)."""
        mine = {p.coord for p in self.players[pid].planets}
        out = []
        for pl in self.players:
            if pl.id == pid:
                continue
            for f in pl.fleets:
                if f.phase == "out" and f.target in mine and f.mission in ("attack", "espionage"):
                    out.append({"mission": f.mission, "from": self.planets[f.origin].coord, "arrives_in": f.t_arrive - self.t,
                                "ships": sum(f.ships.values())})
        return out

    def state(self, pid: int) -> dict:
        pl = self.players[pid]
        for p in pl.planets:
            self.sync(p)
        return {
            "t": self.t, "score": pl.score, "tech": dict(pl.tech), "research": pl.rq,
            "planets": [{"id": p.id, "coord": p.coord, "res": dict(p.res), "rate_per_hour": {r: v * 3600 for r, v in p.rate.items()},
                         "cap": dict(p.cap), "levels": dict(p.lv), "queue": p.bq, "ships": dict(p.ships), "defs": dict(p.defs),
                         "unit_queue": [list(q) for q in p.uq], "fields": (p.used_fields(), p.max_fields())} for p in pl.planets],
            "fleets": [{"id": f.id, "mission": f.mission, "target": f.target, "phase": f.phase, "ships": dict(f.ships),
                        "arrive": f.t_arrive, "return": f.t_return} for f in pl.fleets],
            "debris": {c: dict(d) for c, d in self.debris.items()},
        }


class ActionError(Exception):
    """An action the game would refuse (the real API answers with an error and nothing happens)."""
