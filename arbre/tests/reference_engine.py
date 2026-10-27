"""Implémentation Python lente et littérale d'une partie des features du moteur C (oracle de test).

Elle reproduit la sémantique documentée (fenêtres, plafonds, ordre de parcours du plus récent au plus
ancien, DFS bornée) sans les structures optimisées : listes Python et dictionnaires.
"""

from __future__ import annotations

import math

import numpy as np

F32 = np.float32
NO_MATCH = F32(9.0)


class RefEngine:
    def __init__(self, cfg, fmt_mask: int):
        self.cfg = cfg
        self.mask = fmt_mask
        self.pairs: dict[tuple[int, int], list[int]] = {}
        self.edges: list[dict] = []
        self.out_adj: dict[int, list[int]] = {}
        self.in_adj: dict[int, list[int]] = {}
        self.n_out: dict[int, int] = {}
        self.n_in: dict[int, int] = {}
        self.dist_out: dict[int, int] = {}
        self.dist_in: dict[int, int] = {}
        self.first_t: dict[int, int] = {}
        self.last_out_t: dict[int, int] = {}
        self.ew_cnt_out_l: dict[int, np.float32] = {}
        self.last_upd: dict[int, int] = {}

    # parcours du plus récent au plus ancien, arrêt à la fenêtre ou au plafond
    def _scan(self, lst, t_lo, cap):
        out = []
        for idx in reversed(lst):
            if len(out) >= cap:
                break
            e = self.edges[idx]
            if e["t"] < t_lo:
                break
            out.append(e)
        return out

    def _touch(self, a, t):
        if a not in self.last_upd:
            self.last_upd[a] = t
            return
        dt = t - self.last_upd[a]
        if dt <= 0:
            return
        d = F32(math.exp(-dt / self.cfg.win_long)) if dt < 8 * self.cfg.win_long else F32(0.0)
        self.ew_cnt_out_l[a] = F32(self.ew_cnt_out_l.get(a, F32(0)) * d)
        self.last_upd[a] = t

    def _dfs(self, ctx, node, t_prev, depth, first_amt):
        for e in self._scan(self.out_adj.get(node, []), -(2**31), self.cfg.scan_cap):
            if e["t"] < t_prev:
                return
            ctx["budget"] -= 1
            if ctx["budget"] < 0:
                return
            if ctx["guided"] and (e["usd_out"] < ctx["lo"] or e["usd_out"] > ctx["hi"]):
                continue
            fa = e["usd_out"] if depth == 0 else first_amt
            y = e["dst"]
            if y == ctx["target"]:
                ln = depth + 2
                ctx["found"] += 1
                if ctx["min_len"] == 0 or ln < ctx["min_len"]:
                    ctx["min_len"] = ln
                if ctx["found"] >= 16:
                    ctx["budget"] = -1
                    return
                continue
            if depth + 3 > ctx["max_len"] or y == ctx["start"] or y in ctx["path"]:
                continue
            ctx["path"].append(y)
            self._dfs(ctx, y, e["t"], depth + 1, fa)
            ctx["path"].pop()
            if ctx["budget"] < 0:
                return

    def _cycles(self, a, d, t, x, guided, max_len, budget):
        ctx = {"target": a, "start": d, "lo": F32(x / F32(1.3)), "hi": F32(x * F32(1.3)), "guided": guided,
               "max_len": max_len, "found": 0, "min_len": 0, "budget": budget, "path": [d]}  # fmt: skip
        self._dfs(ctx, d, t - self.cfg.win_long, 0, F32(0))
        return ctx["found"], ctx["min_len"]

    def step(self, t, a, d, xp, xr, fmt) -> dict:
        c = self.cfg
        f = {}
        self._touch(a, t)
        if a != d:
            self._touch(d, t)
        f["s_age"] = t - self.first_t[a] if a in self.first_t else -1
        f["s_dt_out"] = t - self.last_out_t[a] if a in self.last_out_t else -1
        f["s_n_out"] = self.n_out.get(a, 0)
        f["r_n_in"] = self.n_in.get(d, 0)
        f["s_dist_out"] = self.dist_out.get(a, 0)
        f["r_dist_in"] = self.dist_in.get(d, 0)
        f["s_ewc_out_l"] = self.ew_cnt_out_l.get(a, F32(0))
        self_loop = a == d
        in_mask = (not self_loop) and bool((self.mask >> fmt) & 1)
        t_l, t_m = t - c.win_long, t - c.win_mid
        keys = ["p_cnt", "p_dt", "p_rev_cnt", "g_s_outdeg_m", "g_s_outdeg_l", "g_s_outcnt_l", "g_s_indeg_l",
                "g_r_indeg_l", "g_r_outdeg_l", "f_best_err", "f_best_lag", "f_n_match2", "f_n_match10",
                "f_chain_depth", "f_in_sum_l", "f_out_sum_l", "m_2hop", "m_sg", "m_bip", "m_cyc_n",
                "m_cyc_minlen", "m_cyc4_n"]  # fmt: skip
        defaults = {"p_dt": -1, "f_best_err": NO_MATCH, "f_best_lag": -1}
        for k in keys:
            f[k] = defaults.get(k, 0)
        depth, chain_start = 0, t
        if not self_loop:
            p = self.pairs.get((a, d))
            if p:
                f["p_cnt"], f["p_dt"] = p[0], t - p[1]
            pr = self.pairs.get((d, a))
            if pr:
                f["p_rev_cnt"] = pr[0]
            outs = self._scan(self.out_adj.get(a, []), t_l, c.scan_cap)
            so = list(dict.fromkeys(e["dst"] for e in outs))
            f["g_s_outdeg_l"] = len(so)
            f["g_s_outdeg_m"] = len({e["dst"] for e in outs if e["t"] >= t_m})
            f["g_s_outcnt_l"] = len(outs)
            ins = self._scan(self.in_adj.get(a, []), t_l, c.scan_cap)
            si = list(dict.fromkeys(e["src"] for e in ins))
            f["g_s_indeg_l"] = len(si)
            best, best_e, n2, n10 = NO_MATCH, None, 0, 0
            for e in ins:
                if e["usd_in"] > 0:
                    err = F32(abs(F32(xp / e["usd_in"]) - F32(1.0)))
                    if err < best:
                        best, best_e = err, e
                    n2 += err <= F32(0.02)
                    n10 += err <= F32(0.10)
            f["f_best_err"], f["f_n_match2"], f["f_n_match10"] = best, n2, n10
            if best_e is not None:
                f["f_best_lag"] = t - best_e["t"]
                if best <= F32(c.chain_tol):
                    depth = min(255, best_e["depth"] + 1)
                    chain_start = best_e["chain_start"]
                    f["f_chain_depth"] = depth
            f["f_in_sum_l"] = F32(math.log1p(sum(float(e["usd_in"]) for e in ins)))
            f["f_out_sum_l"] = F32(math.log1p(sum(float(e["usd_out"]) for e in outs)))
            rin = self._scan(self.in_adj.get(d, []), t_l, c.scan_cap)
            sd = set(e["src"] for e in rin)
            f["g_r_indeg_l"] = len(sd)
            rout = self._scan(self.out_adj.get(d, []), t_l, c.scan_cap)
            f["g_r_outdeg_l"] = len({e["dst"] for e in rout})
            if in_mask:
                f["m_2hop"] = sum(1 for w in so if w != d and w != a and w in sd)
                sg = 0
                for s0 in si[: c.fan_cap]:
                    seen = set()
                    for e in self._scan(self.out_adj.get(s0, []), t_l, c.fan_cap):
                        w = e["dst"]
                        if w != a and w != d and w in sd and w not in seen:
                            seen.add(w)
                            sg += 1
                f["m_sg"] = sg
                bip = 0
                for r in so[: c.fan_cap]:
                    if r == d:
                        continue
                    seen = set()
                    for e in self._scan(self.in_adj.get(r, []), t_l, c.fan_cap):
                        s2 = e["src"]
                        if s2 != a and s2 != d and s2 in sd and s2 not in seen:
                            seen.add(s2)
                            bip += 1
                f["m_bip"] = bip
                f["m_cyc_n"], f["m_cyc_minlen"] = self._cycles(a, d, t, F32(xp), True, c.cycle_max_len,
                                                               c.cycle_budget)  # fmt: skip
                f["m_cyc4_n"], _ = self._cycles(a, d, t, F32(xp), False, 4, c.cycle_budget // 4)
        # mise à jour
        self.first_t.setdefault(a, t)
        self.first_t.setdefault(d, t)
        if not self_loop:
            self.ew_cnt_out_l[a] = F32(self.ew_cnt_out_l.get(a, F32(0)) + F32(1))
            self.n_out[a] = self.n_out.get(a, 0) + 1
            self.n_in[d] = self.n_in.get(d, 0) + 1
            self.last_out_t[a] = t
            p = self.pairs.get((a, d))
            if p is None:
                self.pairs[(a, d)] = [1, t]
                self.dist_out[a] = self.dist_out.get(a, 0) + 1
                self.dist_in[d] = self.dist_in.get(d, 0) + 1
            else:
                p[0] += 1
                p[1] = t
            if (self.mask >> fmt) & 1:
                self.edges.append({"t": t, "src": a, "dst": d, "usd_out": F32(xp), "usd_in": F32(xr),
                                   "depth": depth, "chain_start": chain_start})  # fmt: skip
                self.out_adj.setdefault(a, []).append(len(self.edges) - 1)
                self.in_adj.setdefault(d, []).append(len(self.edges) - 1)
        return f
