"""A small policy with no framework needed: observation -> hidden layer -> four heads (kind, planet slot, object, amount).
Illegal choices are masked out before choosing, so the policy never wastes a step on something the game would refuse.

NumpyPolicy keeps all its weights in ONE flat vector `theta`, which is what evolution strategies need. (Gradient methods
can use the same masks and heads with PyTorch: see `legal_b`.)
"""
from __future__ import annotations

import numpy as np

from .env import ARGS, MAXP, K_DEBRIS, K_TARGETS, OBS_SIZE, MultiEnv

HEADS = ARGS    # (kind, a, b, c) sizes
NEG = -1e9


def legal_b(env: MultiEnv, pid: int, masks: dict, kind: int, a: int) -> np.ndarray:
    """Which values of `b` are legal for a given kind and planet slot."""
    out = np.zeros(HEADS[2], dtype=bool)
    w = env.world
    pl = w.players[pid]
    if kind == 1:
        out[:masks["build"].shape[1]] = masks["build"][a]
    elif kind == 2:
        out[:len(masks["research"])] = masks["research"]
    elif kind == 3:
        out[:masks["units"].shape[1]] = masks["units"][a]
    elif kind in (4, 5):
        out[:len(env._targets[pid])] = True
    elif kind == 6:
        for pos in range(1, 16):
            out[pos - 1] = w.can_colonise_position(pid, pos)
    elif kind == 7:
        for s in range(len(pl.planets)):
            out[s] = s != a and s < MAXP
    elif kind == 8:
        out[:min(K_DEBRIS, len(w.debris))] = True
    else:
        out[0] = True
    return out


def legal_a(env: MultiEnv, pid: int, masks: dict, kind: int) -> np.ndarray:
    n = len(env.world.players[pid].planets)
    out = np.zeros(HEADS[1], dtype=bool)
    if kind == 1:
        out[:masks["build"].shape[0]] = masks["build"].any(axis=1)
    elif kind == 3:
        out[:masks["units"].shape[0]] = masks["units"].any(axis=1)
    elif kind == 0 or kind == 2:
        out[0] = True
    else:
        out[:min(n, MAXP)] = True
    return out


class NumpyPolicy:
    def __init__(self, hidden: int = 64, theta: np.ndarray | None = None):
        self.hidden = hidden
        self.shapes = [("w1", (OBS_SIZE, hidden)), ("b1", (hidden,)), ("wk", (hidden, HEADS[0])), ("wa", (hidden, HEADS[1])),
                       ("wb", (hidden + HEADS[0], HEADS[2])), ("wc", (hidden, HEADS[3]))]
        self.size = sum(int(np.prod(s)) for _, s in self.shapes)
        self.theta = theta if theta is not None else self.init(np.random.default_rng(0))

    def init(self, rng: np.random.Generator) -> np.ndarray:
        parts = []
        for name, shape in self.shapes:
            parts.append(np.zeros(shape) if name == "b1" else rng.normal(0, 1 / np.sqrt(shape[0]), shape))
        return np.concatenate([p.ravel() for p in parts]).astype(np.float64)

    def _unpack(self, theta: np.ndarray) -> dict:
        key = id(theta)
        hit = getattr(self, "_cache", None)
        if hit is not None and hit[0] == key and hit[1] is theta:
            return hit[2]
        out, i = {}, 0
        for name, shape in self.shapes:
            n = int(np.prod(shape))
            out[name] = theta[i:i + n].reshape(shape)
            i += n
        self._cache = (key, theta, out)
        return out

    def act(self, env: MultiEnv, pid: int, obs: np.ndarray, theta: np.ndarray | None = None, rng: np.random.Generator | None = None,
            explore: float = 0.0) -> tuple:
        """Greedy when rng is None, otherwise sampled. With `explore` > 0 (and an rng) the kind of action is, with that chance,
        drawn uniformly from the legal ones (and then also the planet, object and amount), so rare actions (spy, attack, colonise) show up often enough to be learned."""
        p = self._unpack(self.theta if theta is None else theta)
        masks = env.masks(pid)
        h = np.tanh(obs @ p["w1"] + p["b1"])

        def pick(logits: np.ndarray, ok: np.ndarray, uniform: bool = False) -> int:
            z = np.where(ok, logits, NEG)
            if uniform:   # an exploration step: a uniformly random legal value (flat logits)
                return int(rng.choice(np.flatnonzero(ok)))
            if rng is None:
                return int(np.argmax(z))
            z = z - z.max()
            pr = np.exp(z) * ok
            return int(rng.choice(len(z), p=pr / pr.sum()))

        kind_ok = masks["kind"].copy()
        kind_ok = np.concatenate([kind_ok, np.zeros(HEADS[0] - len(kind_ok), dtype=bool)])
        random_step = explore > 0 and rng is not None and rng.random() < explore
        if random_step:
            kind = int(rng.choice(np.flatnonzero(kind_ok)))
        else:
            kind = pick(h @ p["wk"], kind_ok)
        if kind == 0:
            return (0, 0, 0, 0)
        a = pick(np.zeros(HEADS[1]) if random_step else h @ p["wa"], legal_a(env, pid, masks, kind), random_step)
        onehot = np.zeros(HEADS[0])
        onehot[kind] = 1.0
        b = pick(np.zeros(HEADS[2]) if random_step else np.concatenate([h, onehot]) @ p["wb"], legal_b(env, pid, masks, kind, a), random_step)
        c = pick(np.zeros(HEADS[3]) if random_step else h @ p["wc"], np.ones(HEADS[3], dtype=bool), random_step)
        return (kind, a, b, c)
