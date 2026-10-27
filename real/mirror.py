"""A mirror of the real OGameX universe, as one player sees it, built from the /api/v1 endpoints.

`MirrorWorld` is an ogsim `World` whose state is overwritten from the server on each `refresh`, and whose actions (build, research,
build_units, send_fleet) are real API calls. Because it IS a World, the very same `MultiEnv.observe / masks / act` code that the
agents were trained with produces the observation and interprets the actions: nothing about the game is re-implemented here.

Seat 0 is the player of this client; the other seats are the other players of the highscore (their main planet and score).
"""
from __future__ import annotations

import time

from ogsim import econ
from ogsim.env import MultiEnv
from ogsim.world import ActionError, Config, Fleet, Planet, Player, World
from ogsim.unitdata import UNITS
from ogsim.econdata import OBJECTS

from .api import ApiClient, ApiError

RESOURCES = ("metal", "crystal", "deuterium")
MISSION_FROM_API = {"attack": "attack", "espionage": "espionage", "transport": "transport", "colonisation": "colonisation", "recycle": "recycle"}


class MirrorWorld(World):
    def __init__(self, client: ApiClient, universe: dict):
        sp = universe.get("speed", {})
        cfg = Config(n_players=1, galaxies=int(universe.get("galaxies", 1)), systems=int(universe.get("systems_per_galaxy", 499)),
                     economy_speed=float(sp.get("economy", 1)), fleet_speed=float(sp.get("fleet", 1)), protection_points=0)
        self.cfg = cfg
        self.client = client
        self.t0 = float(universe.get("server_time", time.time()))
        import random
        self.rng = random.Random(0)
        self.t = 0.0
        self.players: list[Player] = []
        self.planets: dict[int, Planet] = {}
        self.by_coord: dict[tuple, Planet] = {}
        self.debris: dict[tuple, dict] = {}
        self.events: list = []
        self._seq = self._fleet_id = self._planet_id = 0
        self.my_id: int | None = None
        self.seat_of: dict[int, int] = {}          # real user id -> seat
        self.api_planet_ids: list[int] = []        # real planet ids of this player, in slot order
        self._slots_max = 1
        self._incoming: list[dict] = []
        self.api_errors: list[str] = []

    # ---- reading the server --------------------------------------------------------------------------------------------
    def refresh(self, highscore: list[dict], status: dict | None = None, fleet: dict | None = None, planet_details: dict | None = None,
                reports: list[dict] | None = None) -> None:
        """Overwrite the mirror from the server. `highscore` is the list of /highscore players (shared between clients)."""
        c = self.client
        status = status or c.get("/status")
        fleet = fleet or c.get("/fleet")
        self.t = float(status["server_time"]) - self.t0
        self.my_id = int(status["player"]["id"])
        # players: me first, then the others by user id
        others = sorted((r for r in highscore if int(r["user_id"]) != self.my_id), key=lambda r: int(r["user_id"]))
        old = self.players
        self.players, self.planets, self.by_coord = [], {}, {}
        self.seat_of = {self.my_id: 0}
        me = Player(0, status["player"].get("username", "me"))
        me.points = float(status["player"].get("points", 0)) * 1000.0
        self.players.append(me)
        for k, r in enumerate(others, start=1):
            pl = Player(k, r.get("username", f"p{k}"))
            pl.points = float(r.get("score", 0)) * 1000.0
            self.seat_of[int(r["user_id"])] = k
            mp = r["main_planet"]
            p = Planet(-k, k, (mp["galaxy"], mp["system"], mp["position"]), 150, (0, 0, 0))
            p.t = self.t
            pl.planets.append(p)
            self.planets[p.id] = p
            self.by_coord[p.coord] = p
            self.players.append(pl)
        # keep what this client learned earlier (espionage reports, battles) across refreshes
        if old and len(old) > 0:
            me.intel, me.battles, me.looted, me.ships_lost = old[0].intel, old[0].battles, old[0].looted, old[0].ships_lost
            for o in old[1:]:
                pass
        self.cfg.n_players = len(self.players)
        # my planets
        me.tech = {k: int(v) for k, v in status["research"]["levels"].items() if int(v) > 0}
        rq = [q for q in status["research"]["queue"]]
        me.rq = None
        if rq:
            q = rq[0]
            me.rq = (self.t + float(q.get("seconds_remaining", 0)), q["machine_name"], int(q.get("target_level", 1)), 0)
        self.api_planet_ids = []
        for sp in status["planets"]:
            if sp.get("type") == "moon":
                continue
            pid = int(sp["id"])
            self.api_planet_ids.append(pid)
            det = (planet_details or {}).get(pid) or c.get(f"/planets/{pid}")
            co = sp["coordinates"]
            lv = {k: int(v) for k, v in det.get("buildings", {}).items()}
            terra = lv.get("terraformer", 0)
            base = int(sp["fields"]["max"]) - (terra * 5 + terra // 2)
            p = Planet(pid, 0, (co["galaxy"], co["system"], co["position"]), base,
                       tuple(float(sp["resources"][r]) for r in RESOURCES))
            p.t = self.t
            p.lv = lv
            p.ships = {n: int(k) for n, k in det.get("units", {}).items() if n in UNITS and UNITS[n]["kind"] == "ship"}
            p.defs = {n: int(k) for n, k in det.get("units", {}).items() if n in UNITS and UNITS[n]["kind"] == "defense"}
            p.rate = {r: float(sp["production_per_hour"][r]) / 3600.0 for r in RESOURCES}
            p.cap = {r: int(sp["storage"][r]) for r in RESOURCES}
            bq = [q for q in sp.get("building_queue", [])]
            p.bq = None
            if bq:
                q = bq[0]
                p.bq = (self.t + float(q.get("seconds_remaining", 0)), q["machine_name"], int(q.get("target_level", 1)))
            p.uq = []
            for q in sp.get("unit_queue", []):
                n_total = max(1, int(q.get("amount", 1)))
                each = float(q.get("duration_seconds", 0)) / n_total if q.get("duration_seconds") else 60.0
                p.uq.append([q["machine_name"], int(q.get("amount_remaining", n_total)), max(1.0, each),
                             self.t + float(q.get("seconds_until_next_unit", each))])
            me.planets.append(p)
            self.planets[pid] = p
            self.by_coord[p.coord] = p
        # fleets and what is coming at me
        self._slots_max = int(fleet["fleet_slots"]["max"])
        mine_coords = {p.coord for p in me.planets}
        me.fleets, self._incoming = [], []
        for m in fleet["missions"]:
            to = (m["to"]["galaxy"], m["to"]["system"], m["to"]["position"])
            frm = (m["from"]["galaxy"], m["from"]["system"], m["from"]["position"])
            if m.get("is_mine"):
                me.fleets.append(Fleet(int(m["id"]), 0, int(m["from"].get("planet_id") or 0), to, MISSION_FROM_API.get(m["mission"], m["mission"]),
                                       dict(m.get("ships", {})), dict(m.get("resources", {})), self.t - 1, self.t + float(m["seconds_remaining"]),
                                       self.t + float(m["seconds_remaining"]), "back" if m.get("is_return") else "out"))
            elif to in mine_coords and m["mission"] in ("attack", "espionage") and not m.get("is_return"):
                self._incoming.append({"mission": m["mission"], "from": frm, "arrives_in": float(m["seconds_remaining"]), "ships": 1})
        for rep in reports or []:
            self._add_report(rep)

    def _add_report(self, rep: dict) -> None:
        """Turn an API espionage report into the sim's intel entry."""
        t = rep["target"]
        seat = self.seat_of.get(int(t.get("owner_id") or -1))
        if seat is None:
            return
        entry = {"t": float(rep.get("t", self.t)), "target": (t["galaxy"], t["system"], t["position"]), "owner": seat,
                 "resources": {r: int((rep.get("resources") or {}).get(r, 0)) for r in RESOURCES}}
        if rep.get("ships") is not None:
            entry["ships"] = {n: int(k) for n, k in rep["ships"].items() if n in UNITS}
        if rep.get("defense") is not None:
            entry["defense"] = {n: int(k) for n, k in rep["defense"].items() if n in UNITS}
        if rep.get("buildings"):
            entry["buildings"] = dict(rep["buildings"])
        if rep.get("research"):
            entry["research"] = dict(rep["research"])
        self.players[0].intel.append(entry)

    # ---- the pieces of World that differ for a real game --------------------------------------------------------------
    def fleet_slots(self, pid: int) -> int:
        return self._slots_max

    def incoming(self, pid: int) -> list[dict]:
        return list(self._incoming) if pid == 0 else []

    def max_planets(self, pid: int) -> int:
        return super().max_planets(pid)

    def advance(self, seconds: float) -> None:      # real time passes on its own
        raise RuntimeError("the real universe cannot be advanced from here")

    # ---- actions are API calls ---------------------------------------------------------------------------------------
    def _post(self, path: str, body: dict):
        try:
            return self.client.post(path, body)
        except ApiError as e:
            self.api_errors.append(f"{e.request_line}: {e.code}")
            raise ActionError(f"{e.code}: {e.message}")

    def build(self, pid: int, planet: int, name: str) -> float:
        p = self._planet_of(pid, planet)
        self._post(f"/planets/{p.id}/buildings", {"machine_name": name})
        return 0.0

    def research(self, pid: int, name: str, planet: int = 0) -> float:
        p = self._planet_of(pid, planet)
        self._post(f"/planets/{p.id}/research", {"machine_name": name})
        return 0.0

    def build_units(self, pid: int, planet: int, name: str, amount: int) -> float:
        p = self._planet_of(pid, planet)
        self._post(f"/planets/{p.id}/units", {"machine_name": name, "amount": int(amount)})
        return 0.0

    def send_fleet(self, pid: int, planet: int, mission: str, target: tuple, ships: dict, cargo: dict | None = None,
                   speed_pct: float = 100.0) -> int:
        p = self._planet_of(pid, planet)
        body = {"mission": mission, "galaxy": target[0], "system": target[1], "position": target[2],
                "type": "debris_field" if mission == "recycle" else "planet",
                "ships": {n: int(k) for n, k in ships.items() if int(k) > 0},
                "resources": {r: int((cargo or {}).get(r, 0)) for r in RESOURCES},
                "speed_percent": int(round(speed_pct / 5.0) * 5)}
        data = self._post(f"/planets/{p.id}/fleet", body)
        return int(data["mission"]["id"])


class RealEnv(MultiEnv):
    """MultiEnv over a real universe: one agent (seat 0). Use `observe()`, `masks()` and `act()` exactly as in training."""

    def __init__(self, client: ApiClient, decision_seconds: float = 600.0, horizon_seconds: float = 7 * 86400.0):
        self.client = client
        universe = client.get("/universe")
        super().__init__(Config(n_players=1), decision_seconds=decision_seconds, horizon_seconds=horizon_seconds)
        self.world = MirrorWorld(client, universe)
        self.cfg = self.world.cfg
        self._last_score = [0.0]
        self._seen_reports: set[int] = set()

    @property
    def n(self) -> int:
        return self.cfg.n_players

    def fetch_highscore(self) -> list[dict]:
        out, page = [], 1
        while True:
            d = self.client.get(f"/highscore?per_page=100&page={page}")
            out += d["players"]
            if page >= d["total_pages"]:
                return out
            page += 1

    def new_reports(self) -> list[dict]:
        """Espionage reports that arrived since the last call (each fetched in full once)."""
        found = []
        try:
            lst = self.client.get("/messages?category=espionage&per_page=50")
        except ApiError:
            return found
        for m in lst["messages"]:
            if m.get("has_espionage_report") and m["id"] not in self._seen_reports:
                self._seen_reports.add(m["id"])
                try:
                    full = self.client.get(f"/messages/{m['id']}")["message"]
                except ApiError:
                    continue
                rep = dict(full.get("espionage_report") or {})
                if rep:
                    rep["t"] = (full.get("created_at") or 0) - self.world.t0
                    found.append(rep)
        return found

    def refresh(self, highscore: list[dict] | None = None) -> np.ndarray:
        """Read the server, rebuild the mirror and return the observation. Call once per decision."""
        hs = highscore if highscore is not None else self.fetch_highscore()
        self.world.refresh(hs, reports=self.new_reports())
        self._targets = {}
        self._refresh_targets()
        return self.observe(0)

    def step(self, actions):                       # a real game is not stepped; use act()
        raise RuntimeError("use act() and refresh() against a real server")


import numpy as np  # noqa: E402  (kept at the end: only used in a type hint)
