import random
import unittest

from ogsim import battle as b


class BattleTest(unittest.TestCase):
    def test_empty_planet_is_looted_by_half(self):
        o = b.fight({"large_cargo": 4, "light_fighter": 20}, {}, {"metal": 10000, "crystal": 4000, "deuterium": 2000}, random.Random(1))
        self.assertTrue(o.attacker_won)
        self.assertEqual(o.loot, {"metal": 5000.0, "crystal": 2000.0, "deuterium": 1000.0})
        self.assertEqual(sum(o.debris.values()), 0)

    def test_split_loot_even_thirds_then_leftover(self):
        got = b.split_loot({"metal": 9000, "crystal": 3000, "deuterium": 0}, 6000)
        self.assertAlmostEqual(got["metal"], 3000, delta=1)
        self.assertAlmostEqual(got["crystal"], 3000, delta=1)
        self.assertEqual(got["deuterium"], 0)

    def test_attackers_lose_and_leave_debris(self):
        o = b.fight({"light_fighter": 10}, {"rocket_launcher": 60}, {"metal": 1e6, "crystal": 1e6, "deuterium": 1e6}, random.Random(2))
        self.assertFalse(o.attacker_won)
        self.assertEqual(sum(o.loot.values()), 0)
        self.assertAlmostEqual(o.debris["metal"], 0.3 * 3000 * o.attacker_lost["light_fighter"], delta=1)
        self.assertEqual(o.debris["crystal"], 0.3 * 1000 * o.attacker_lost["light_fighter"])

    def test_defence_rebuilt_70_percent(self):
        o = b.fight({"cruiser": 40}, {"rocket_launcher": 30}, {}, random.Random(3))
        self.assertTrue(o.attacker_won)
        self.assertGreaterEqual(o.defender_left["rocket_launcher"], 21)
        self.assertEqual(o.defender_left["rocket_launcher"] + o.defender_lost["rocket_launcher"], 30)

    def test_cargo_limits_loot(self):
        o = b.fight({"small_cargo": 2, "cruiser": 10}, {}, {"metal": 1e6, "crystal": 1e6, "deuterium": 1e6}, random.Random(4))
        self.assertLessEqual(sum(o.loot.values()), 2 * 5000 + 10 * 800 + 1)

    def test_old_estimate_agrees(self):   # 13 light fighters lose against 14 rocket launchers (real report of the owner's server)
        o = b.fight({"light_fighter": 13}, {"rocket_launcher": 14}, {}, random.Random(5))
        self.assertFalse(o.attacker_won)

    def test_big_fight_is_scaled(self):
        o = b.fight({"cruiser": 20000}, {"rocket_launcher": 20000}, {}, random.Random(6))
        self.assertTrue(o.attacker_won)
        self.assertLessEqual(sum(o.defender_left.values()) + sum(o.defender_lost.values()), 20000)

if __name__ == "__main__":
    unittest.main()
