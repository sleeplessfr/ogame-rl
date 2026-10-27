"""PPO in plain numpy (no PyTorch needed): one hidden layer shared by four masked action heads and a value head.

The action is chosen in four steps, like the environment expects: kind, then planet slot, then object, then amount. The log-probability
of an action is the sum over the steps that were used (a plain `noop` uses only the first). Illegal choices are masked out, so the
policy only ever chooses among legal ones. Backpropagation is written by hand and checked against finite differences in the tests.
"""
from __future__ import annotations

import numpy as np

from .env import ARGS, OBS_SIZE, MultiEnv
from .policy import legal_a, legal_b

NK, NA, NB, NC = ARGS
NEG = -1e9


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class Net:
    def __init__(self, hidden: int = 128, seed: int = 0):
        self.H = hidden
        rng = np.random.default_rng(seed)
        H = hidden
        self.p = {
            "W1": rng.normal(0, 1 / np.sqrt(OBS_SIZE), (OBS_SIZE, H)), "b1": np.zeros(H),
            "Wk": rng.normal(0, 0.01, (H, NK)), "bk": np.zeros(NK),
            "Wa": rng.normal(0, 0.01, (H, NA)), "ba": np.zeros(NA),
            "Wb": rng.normal(0, 0.01, (H + NK, NB)), "bb": np.zeros(NB),
            "Wc": rng.normal(0, 0.01, (H, NC)), "bc": np.zeros(NC),
            "wv": rng.normal(0, 0.01, (H,)), "bv": np.zeros(1),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ---- forward ----------------------------------------------------------------------------------------------------
    def trunk(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.p["W1"] + self.p["b1"])

    def value(self, h: np.ndarray) -> np.ndarray:
        return h @ self.p["wv"] + self.p["bv"][0]

    def logits_k(self, h): return h @ self.p["Wk"] + self.p["bk"]
    def logits_a(self, h): return h @ self.p["Wa"] + self.p["ba"]
    def logits_c(self, h): return h @ self.p["Wc"] + self.p["bc"]

    def logits_b(self, h, kind):
        oh = np.zeros((len(h), NK))
        oh[np.arange(len(h)), kind] = 1.0
        return np.concatenate([h, oh], axis=1) @ self.p["Wb"] + self.p["bb"], oh

    # ---- acting ------------------------------------------------------------------------------------------------------
    def act(self, env: MultiEnv, pids: list[int], obs: dict[int, np.ndarray], rng: np.random.Generator, greedy: bool = False):
        """Chooses an action for each player in `pids`. Returns (actions, records) where records feed `update`."""
        X = np.stack([obs[i] for i in pids]).astype(np.float64)
        h = self.trunk(X)
        v = self.value(h)
        masks = [env.masks(i) for i in pids]
        km = np.zeros((len(pids), NK), dtype=bool)
        for r, m in enumerate(masks):
            km[r, :len(m["kind"])] = m["kind"]
        pk = _softmax(np.where(km, self.logits_k(h), NEG))
        kind = np.array([int(np.argmax(pk[r])) if greedy else int(rng.choice(NK, p=pk[r] / pk[r].sum())) for r in range(len(pids))])
        am = np.zeros((len(pids), NA), dtype=bool)
        for r, m in enumerate(masks):
            if kind[r] > 0:
                am[r] = legal_a(env, pids[r], m, int(kind[r]))
        pa = _softmax(np.where(am, self.logits_a(h), NEG))
        a = np.array([0 if kind[r] == 0 else (int(np.argmax(pa[r])) if greedy else int(rng.choice(NA, p=pa[r] / pa[r].sum()))) for r in range(len(pids))])
        bm = np.zeros((len(pids), NB), dtype=bool)
        for r, m in enumerate(masks):
            if kind[r] > 0:
                bm[r] = legal_b(env, pids[r], m, int(kind[r]), int(a[r]))
        lb, _ = self.logits_b(h, kind)
        pb = _softmax(np.where(bm, lb, NEG))
        b = np.array([0 if kind[r] == 0 else (int(np.argmax(pb[r])) if greedy else int(rng.choice(NB, p=pb[r] / pb[r].sum()))) for r in range(len(pids))])
        pc = _softmax(self.logits_c(h))
        c = np.array([0 if kind[r] == 0 else (int(np.argmax(pc[r])) if greedy else int(rng.choice(NC, p=pc[r]))) for r in range(len(pids))])
        idx = np.arange(len(pids))
        used = kind > 0
        logp = np.log(pk[idx, kind] + 1e-12) + used * (np.log(pa[idx, a] + 1e-12) + np.log(pb[idx, b] + 1e-12) + np.log(pc[idx, c] + 1e-12))
        actions = {pid: (int(kind[r]), int(a[r]), int(b[r]), int(c[r])) for r, pid in enumerate(pids)}
        rec = {"obs": X, "km": km, "am": am, "bm": bm, "kind": kind, "a": a, "b": b, "c": c, "logp": logp, "v": v}
        return actions, rec

    # ---- learning ----------------------------------------------------------------------------------------------------
    def loss_grad(self, B: dict, clip: float = 0.2, vf: float = 0.5, ent: float = 0.01):
        """PPO loss over a batch and its gradient (by hand). B has obs, km, am, bm, kind, a, b, c, logp, adv, ret."""
        p, N = self.p, len(B["obs"])
        X = B["obs"]
        h = self.trunk(X)
        v = self.value(h)
        idx = np.arange(N)
        used = (B["kind"] > 0).astype(np.float64)
        Zk = np.where(B["km"], self.logits_k(h), NEG)
        Za = np.where(B["am"], self.logits_a(h), NEG)
        Lb, oh = self.logits_b(h, B["kind"])
        Zb = np.where(B["bm"], Lb, NEG)
        Zc = self.logits_c(h)
        Pk, Pa, Pb, Pc = _softmax(Zk), _softmax(Za), _softmax(Zb), _softmax(Zc)
        logp = (np.log(Pk[idx, B["kind"]] + 1e-12) + used * (np.log(Pa[idx, B["a"]] + 1e-12) + np.log(Pb[idx, B["b"]] + 1e-12)
                                                          + np.log(Pc[idx, B["c"]] + 1e-12)))
        ratio = np.exp(np.clip(logp - B["logp"], -20, 20))
        adv = B["adv"]
        unclipped = ratio * adv
        clipped = np.clip(ratio, 1 - clip, 1 + clip) * adv
        pol_loss = -np.minimum(unclipped, clipped).mean()
        use_grad = (unclipped <= clipped) | ((ratio > 1 - clip) & (ratio < 1 + clip))
        # d loss / d logp  (zero where the clipped branch is active)
        g = np.where(use_grad, -adv * ratio, 0.0) / N

        def ent_of(P):
            return -(P * np.log(P + 1e-12)).sum(axis=1)
        Hk, Ha, Hb, Hc = ent_of(Pk), ent_of(Pa), ent_of(Pb), ent_of(Pc)
        entropy = (Hk + used * (Ha + Hb + Hc)).mean()
        v_loss = 0.5 * ((v - B["ret"]) ** 2).mean()
        total = pol_loss + vf * v_loss - ent * entropy

        def dz(P, onehot_idx, weight, H):
            """gradient of (g * logp_part - ent * H_part) with respect to the logits of that part"""
            oh_ = np.zeros_like(P)
            oh_[idx, onehot_idx] = 1.0
            d = g[:, None] * (oh_ - P)                                            # policy part
            d += (ent / N) * P * (np.log(P + 1e-12) + H[:, None])                  # entropy bonus (loss = -ent * H)
            return d * weight[:, None]
        ones = np.ones(N)
        dZk = dz(Pk, B["kind"], ones, Hk)
        dZa = dz(Pa, B["a"], used, Ha)
        dZb = dz(Pb, B["b"], used, Hb)
        dZc = dz(Pc, B["c"], used, Hc)
        dv = vf * (v - B["ret"]) / N
        G = {}
        G["Wk"], G["bk"] = h.T @ dZk, dZk.sum(0)
        G["Wa"], G["ba"] = h.T @ dZa, dZa.sum(0)
        hb = np.concatenate([h, oh], axis=1)
        G["Wb"], G["bb"] = hb.T @ dZb, dZb.sum(0)
        G["Wc"], G["bc"] = h.T @ dZc, dZc.sum(0)
        G["wv"], G["bv"] = h.T @ dv, np.array([dv.sum()])
        dh = dZk @ p["Wk"].T + dZa @ p["Wa"].T + (dZb @ p["Wb"].T)[:, :self.H] + dZc @ p["Wc"].T + np.outer(dv, p["wv"])
        dpre = dh * (1 - h ** 2)
        G["W1"], G["b1"] = X.T @ dpre, dpre.sum(0)
        stats = {"loss": float(total), "pol": float(pol_loss), "vf": float(v_loss), "ent": float(entropy),
                 "kl": float(np.mean(B["logp"] - logp)), "clipfrac": float(np.mean(~use_grad))}
        return total, G, stats

    def adam(self, G: dict, lr: float = 3e-4, max_norm: float = 0.5) -> None:
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in G.values()))
        scale = min(1.0, max_norm / (norm + 1e-8))
        self.t += 1
        for k, g in G.items():
            g = g * scale
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def update(self, B: dict, epochs: int = 4, minibatch: int = 4096, lr: float = 3e-4, rng: np.random.Generator | None = None, **kw) -> dict:
        rng = rng or np.random.default_rng(0)
        n = len(B["obs"])
        stats = {}
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, minibatch):
                sel = order[s:s + minibatch]
                mb = {k: v[sel] for k, v in B.items()}
                _, G, stats = self.loss_grad(mb, **kw)
                self.adam(G, lr)
        return stats

    # ---- saving ------------------------------------------------------------------------------------------------------
    def state(self) -> dict:
        return {k: v.copy() for k, v in self.p.items()}

    def load(self, state: dict) -> None:
        for k in self.p:
            self.p[k] = np.array(state[k], dtype=np.float64)


def gae(r: np.ndarray, v: np.ndarray, gamma: float = 0.995, lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns of one seat's trajectory (the episode ends after the last step; nothing is bootstrapped past it)."""
    T = len(r)
    adv = np.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        nv = v[t + 1] if t + 1 < T else 0.0
        delta = r[t] + gamma * nv - v[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + v
