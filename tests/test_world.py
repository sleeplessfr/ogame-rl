import unittest

from ogsim.world import ActionError, Config, World


def rich(w, pid=0, n=10_000_000):
    for p in w.players[pid].planets:
        p.res = {"metal": float(n), "crystal": float(n), "deuterium": float(n)}


class WorldTest(unittest.TestCase):
    def setUp(self):
        self.w = World(Config(n_players=4, seed=3, systems=20))

    def test_production_grows_stock_over_time(self):
        p = self.w.players[0].planets[0]
        m0 = p.res["metal"]
        self.w.advance(3600)
        self.assertGreater(p.res["metal"], m0 + 1000)   # basic income at x100: 30*100/h metal plus bonuses

    def test_building_takes_time_costs_and_levels_up(self):
        w = self.w
        p = w.players[0].planets[0]
        w.advance(7200)
        secs = w.build(0, 0, "metal_mine")
        self.assertGreater(secs, 0)
        with self.assertRaises(ActionError):
            w.build(0, 0, "crystal_mine")             # queue busy
        w.advance(secs + 1)
        self.assertEqual(p.lv["metal_mine"], 1)
        self.assertGreater(w.players[0].score, 0)

    def test_requirements_and_resources_are_enforced(self):
        w = self.w
        with self.assertRaises(ActionError):
            w.build(0, 0, "research_lab") if False else w.research(0, "energy_technology")   # no lab
        with self.assertRaises(ActionError):
            w.build(0, 0, "fusion_plant")              # needs deuterium synthesizer and energy tech
        w.players[0].planets[0].res = {"metal": 0.0, "crystal": 0.0, "deuterium": 0.0}
        with self.assertRaises(ActionError):
            w.build(0, 0, "metal_mine")

    def test_research_needs_lab_and_applies_to_all(self):
        w = self.w
        rich(w)
        w.build(0, 0, "research_lab")
        w.advance(100000)
        secs = w.research(0, "energy_technology")
        with self.assertRaises(ActionError):
            w.research(0, "espionage_technology")      # one at a time
        w.advance(secs + 1)
        self.assertEqual(w.players[0].tech["energy_technology"], 1)

    def test_units_come_out_one_by_one(self):
        w = self.w
        rich(w)
        for name in ("robot_factory", "robot_factory", "shipyard"):
            w.build(0, 0, name)
            w.advance(100000)
        total = w.build_units(0, 0, "rocket_launcher", 5)
        w.advance(total / 5 * 2.5)
        p = w.players[0].planets[0]
        self.assertEqual(p.defs.get("rocket_launcher", 0), 2)
        w.advance(total)
        self.assertEqual(p.defs.get("rocket_launcher", 0), 5)

    def _army(self, w, pid, ships):
        p = w.players[pid].planets[0]
        for n, k in ships.items():
            p.ships[n] = p.ships.get(n, 0) + k
        rich(w, pid)
        w.players[pid].tech["computer_technology"] = 5

    def test_spy_report_shows_resources_and_more_with_probes(self):
        w = self.w
        self._army(w, 0, {"espionage_probe": 10})
        target = w.players[1].planets[0]
        w.send_fleet(0, 0, "espionage", target.coord, {"espionage_probe": 10})
        w.advance(100000)
        rep = w.players[0].intel[-1]
        self.assertIn("resources", rep)
        self.assertIn("defense", rep)                  # 10 probes, equal tech: defence is shown (3+ needed)
        self.assertEqual(w.players[0].planets[0].ships["espionage_probe"], 10)   # probes came home

    def test_one_probe_sees_no_defence(self):
        w = self.w
        self._army(w, 0, {"espionage_probe": 1})
        w.send_fleet(0, 0, "espionage", w.players[1].planets[0].coord, {"espionage_probe": 1})
        w.advance(100000)
        self.assertNotIn("defense", w.players[0].intel[-1])

    def test_attack_loots_and_returns_with_cargo(self):
        w = self.w
        self._army(w, 0, {"cruiser": 30, "large_cargo": 40})
        victim = w.players[1].planets[0]
        victim.res = {"metal": 200000.0, "crystal": 100000.0, "deuterium": 50000.0}
        before = w.players[0].planets[0].res["metal"]
        w.send_fleet(0, 0, "attack", victim.coord, {"cruiser": 30, "large_cargo": 40})
        w.advance(400000)
        self.assertGreater(w.players[0].looted, 100000)
        self.assertEqual(len(w.players[0].fleets), 0)
        self.assertGreater(w.players[0].planets[0].res["metal"], before)
        self.assertLess(victim.res["metal"], 150000)
        self.assertTrue(w.players[0].battles and w.players[1].battles)

    def test_failed_attack_leaves_debris_that_recyclers_collect(self):
        w = self.w
        self._army(w, 0, {"light_fighter": 10, "recycler": 5})
        victim = w.players[1].planets[0]
        victim.defs["rocket_launcher"] = 100
        w.send_fleet(0, 0, "attack", victim.coord, {"light_fighter": 10})
        w.advance(100000)
        self.assertIn(victim.coord, w.debris)
        d = dict(w.debris[victim.coord])
        w.send_fleet(0, 0, "recycle", victim.coord, {"recycler": 5})
        w.advance(100000)
        self.assertGreater(w.players[0].planets[0].res["metal"], 10_000_000)
        self.assertLess(sum(w.debris.get(victim.coord, {"metal": 0, "crystal": 0}).values()), sum(d.values()))

    def test_colonisation_makes_a_planet(self):
        w = self.w
        w.players[0].tech["astrophysics"] = 1
        self._army(w, 0, {"colony_ship": 1})
        free = w._free_coord(4, 12)
        w.send_fleet(0, 0, "colonisation", free, {"colony_ship": 1})
        w.advance(200000)
        self.assertEqual(len(w.players[0].planets), 2)
        self.assertEqual(w.players[0].planets[1].coord, free)

    def test_fleet_slots_and_fuel(self):
        w = self.w
        self._army(w, 0, {"small_cargo": 10})
        w.players[0].tech["computer_technology"] = 0
        target = w.players[1].planets[0].coord
        w.send_fleet(0, 0, "transport", target, {"small_cargo": 1}, {"metal": 100})
        with self.assertRaises(ActionError):
            w.send_fleet(0, 0, "transport", target, {"small_cargo": 1})
        w.players[0].planets[0].res["deuterium"] = 0.0
        w.players[0].tech["computer_technology"] = 3
        with self.assertRaises(ActionError):
            w.send_fleet(0, 0, "transport", target, {"small_cargo": 1})

    def test_protection(self):
        w = World(Config(n_players=2, seed=1, protection_points=5000))
        self._army(w, 0, {"cruiser": 5})
        w.players[0].points = 50_000_000       # score 50,000
        with self.assertRaises(ActionError):
            w.send_fleet(0, 0, "attack", w.players[1].planets[0].coord, {"cruiser": 5})

    def test_speed(self):
        import time
        w = World(Config(n_players=200, seed=5))
        t0 = time.time()
        for day in range(30):
            w.advance(86400)
        self.assertLess(time.time() - t0, 5)


if __name__ == "__main__":
    unittest.main()
