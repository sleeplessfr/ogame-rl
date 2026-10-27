# ogame-rl

A very fast simulator of an OGame universe (the rules of [OGameX](https://github.com/lanedirt/ogamex)), built to train
reinforcement-learning agents that start from nothing: no hand-written strategy, only the game's legal actions.

* `ogsim/world.py` the universe: planets, production, building and research queues, ships and defence, fleets (espionage, attack,
  transport, colonisation, recycle), battles, loot, debris, scores. Event driven: a game day costs only the actions taken.
* `ogsim/env.py` the training interface: observation vector, 4-integer action, legal-action masks, reward = change in score.
* `ogsim/bots.py` random and plain-builder opponents / yardsticks.
* `ogsim/econdata.py`, `ogsim/unitdata.py` generated from the game's PHP source (`tools/gen_*.py --check` fails when the game changes).
* `ogsim/battle.py` the battle engine (6 rounds, rapid fire, shields, explosion) with the game's loot / debris / defence-rebuild rules.

Run the tests: `python -m unittest discover -s tests`.

Left out on purpose: moons, missiles, expeditions, ACS, officers, items, the merchant, vacation mode.
