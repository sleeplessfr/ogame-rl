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

## Playing on a real OGameX server (`real/`, `run_real.py`)
`real/mirror.py` rebuilds, from the server's `/api/v1` endpoints, an ogsim `World` as one player sees it (his planets, queues, research, fleets, hostile
fleets coming, the highscore with every player's main planet, espionage reports). Because it is a World, the trained observation, legal masks and
action meanings come from the same code as in training; build / research / units / fleet actions are real API calls. `tests/test_real.py`
checks that the observation (404 numbers) and masks read through the API JSON equal the simulator's. `real/fake_server.py` is an in-process
stand-in for the API for testing. Not mirrored yet: debris fields (the API has none), battle counters (looted, ships lost).

    python run_real.py --fake --weights runs/ppo4.npz --agents 8 --days 1          # check the loop without a server
    python run_real.py --base-url http://localhost/api/v1 --weights runs/ppo4.npz --agents 20 --dt 600
