"""An in-process stand-in for the OGameX /api/v1 API, backed by an ogsim World. It speaks the same JSON as the real controllers
(read from the server's PHP), so the adapter can be tested end to end without a running server. Same interface as ApiClient."""
from __future__ import annotations

from ogsim.econdata import OBJECTS
from ogsim.env import BUILDINGS
from ogsim.unitdata import UNITS
from ogsim.world import ActionError, World

from .api import ApiError

RES = ("metal", "crystal", "deuterium")


class FakeClient:
    def __init__(self, world: World, seat: int, t0: float = 1_800_000_000.0):
        self.w, self.seat, self.t0 = world, seat, t0
        self.log: list[tuple] = []

    # -- helpers
    def _err(self, code: str, msg: str, status: int = 422):
        e = ApiError(status, code, msg)
        e.request_line = "fake"
        raise e

    def _planet(self, pid: int):
        for i, p in enumerate(self.w.players[self.seat].planets):
            if p.id == pid:
                return i, p
        self._err("planet_not_found", "no such planet", 404)

    def _q(self, name, level, left, active=True):
        return {"queue_id": 1, "object_id": 1, "machine_name": name, "title": name, "active": active, "seconds_remaining": int(left),
                "duration_seconds": int(left), "target_level": level}

    def _planet_summary(self, p):
        self.w.sync(p)
        return {"id": p.id, "name": "p", "type": "planet", "coordinates": dict(zip(("galaxy", "system", "position"), p.coord)),
                "fields": {"used": p.used_fields(), "max": p.max_fields()},
                "resources": {r: int(p.res[r]) for r in RES}, "storage": {r: int(p.cap[r]) for r in RES},
                "production_per_hour": {r: int(p.rate[r] * 3600) for r in RES}, "energy": 0}

    def _queues(self, p):
        w = self.w
        bq = [self._q(p.bq[1], p.bq[2], p.bq[0] - w.t)] if p.bq else []
        uq = [{"queue_id": 2, "object_id": 1, "machine_name": q[0], "title": q[0], "active": i == 0, "seconds_remaining": 0,
               "duration_seconds": int(q[2] * q[1]), "amount": q[1], "amount_remaining": q[1], "seconds_until_next_unit": int(max(0, q[3] - w.t))}
              for i, q in enumerate(p.uq)]
        return bq, uq

    # -- the client interface
    def get(self, path: str):
        self.log.append(("GET", path))
        w, me = self.w, self.w.players[self.seat]
        path, _, query = path.partition("?")
        if path == "/universe":
            return {"name": "fake", "galaxies": w.cfg.galaxies, "systems_per_galaxy": w.cfg.systems, "planet_positions": 15, "expedition_position": 16,
                    "speed": {"economy": w.cfg.economy_speed, "research": w.cfg.economy_speed, "fleet": w.cfg.fleet_speed}, "server_time": int(self.t0)}
        if path == "/status":
            planets = []
            for p in me.planets:
                bq, uq = self._queues(p)
                planets.append(self._planet_summary(p) | {"building_queue": bq, "unit_queue": uq})
            rq = [self._q(me.rq[1], me.rq[2], me.rq[0] - w.t)] if me.rq else []
            return {"server_time": int(self.t0 + w.t), "player": {"id": 1000 + self.seat, "username": f"p{self.seat}", "points": me.score},
                    "planets": planets, "research": {"levels": {t: me.tech.get(t, 0) for t in self._techs()}, "queue": rq},
                    "fleet_slots": {"used": len(me.fleets), "max": w.fleet_slots(self.seat)}, "unread_messages": 0}
        if path.startswith("/planets/"):
            _, p = self._planet(int(path.split("/")[2]))
            units = {**p.ships, **p.defs}
            return self._planet_summary(p) | {"buildings": {k: v for k, v in p.lv.items() if v}, "units": {k: v for k, v in units.items() if v}}
        if path == "/fleet":
            ms = []
            for f in me.fleets:
                o = w.planets[f.origin]
                ms.append({"id": f.id, "mission": f.mission, "is_mine": True, "is_return": f.phase == "back",
                           "from": dict(zip(("galaxy", "system", "position"), o.coord)) | {"planet_id": o.id},
                           "to": dict(zip(("galaxy", "system", "position"), f.target)) | {"planet_id": None},
                           "seconds_remaining": int((f.t_arrive if f.phase == "out" else f.t_return) - w.t), "ships": dict(f.ships), "resources": dict(f.cargo)})
            for x in w.incoming(self.seat):
                ms.append({"id": 9000, "mission": x["mission"], "is_mine": False, "is_return": False,
                           "from": dict(zip(("galaxy", "system", "position"), x["from"])) | {"planet_id": None},
                           "to": dict(zip(("galaxy", "system", "position"), me.planets[0].coord)) | {"planet_id": me.planets[0].id},
                           "seconds_remaining": int(x["arrives_in"])})
            return {"fleet_slots": {"used": len(me.fleets), "max": w.fleet_slots(self.seat)}, "missions": ms}
        if path == "/highscore":
            rows = [{"rank": 1, "user_id": 1000 + pl.id, "username": f"p{pl.id}", "score": pl.score, "is_me": pl.id == self.seat,
                     "main_planet": dict(zip(("galaxy", "system", "position"), pl.planets[0].coord))} for pl in w.players]
            return {"players": rows, "page": 1, "total_pages": 1, "total_players": len(rows)}
        if path == "/messages":
            return {"messages": [{"id": 100 + i, "has_espionage_report": True} for i, r in enumerate(me.intel)], "total_pages": 1}
        if path.startswith("/messages/"):
            i = int(path.split("/")[2]) - 100
            r = me.intel[i]
            t = dict(zip(("galaxy", "system", "position"), r["target"])) | {"owner_id": 1000 + r["owner"]}
            return {"message": {"id": 100 + i, "created_at": int(self.t0 + r["t"]),
                                "espionage_report": {"target": t, "resources": r["resources"], "ships": r.get("ships"),
                                                     "defense": r.get("defense"), "buildings": r.get("buildings"), "research": r.get("research")}}}
        self._err("not_found", path, 404)

    def _techs(self):
        from ogsim.env import TECHS
        return TECHS

    def post(self, path: str, body: dict | None = None):
        self.log.append(("POST", path, body))
        w, body = self.w, body or {}
        parts = path.strip("/").split("/")
        slot, p = self._planet(int(parts[1]))
        try:
            if parts[2] == "buildings":
                w.build(self.seat, slot, body["machine_name"])
            elif parts[2] == "research":
                w.research(self.seat, body["machine_name"], slot)
            elif parts[2] == "units":
                w.build_units(self.seat, slot, body["machine_name"], int(body["amount"]))
                return {"ordered": int(body["amount"])}
            elif parts[2] == "fleet":
                tgt = (body["galaxy"], body["system"], body["position"])
                fid = w.send_fleet(self.seat, slot, body["mission"], tgt, body["ships"], body.get("resources"), float(body.get("speed_percent", 100)))
                return {"mission": {"id": fid}}
            else:
                self._err("not_found", path, 404)
        except ActionError as e:
            self._err("mission_rejected", str(e))
        return {"queue": []}
