"""Tanker Run solver: starter -> Clarke-Wright savings -> local search
(or-opt/relocate, swap, 2-opt*, 2-opt) -> iterated ruin & recreate.
Submits every new best, so early checkpoints (5%/20%) are already strong."""

import random
import time

from adapter import Solver
from adapters.starter import StarterSolver
from data import distance_matrix

SAFETY_S = 0.25


def rlen(d, r):
    if not r:
        return 0
    t = d[0][r[0]] + d[r[-1]][0]
    for k in range(len(r) - 1):
        t += d[r[k]][r[k + 1]]
    return t


def cost(d, routes):
    return sum(rlen(d, r) for r in routes)


def savings(d, instance):
    n, dem, cap = instance.size, instance.demand, instance.capacity
    routes = {v: [v] for v in range(1, n + 1)}
    load = {v: dem[v] for v in range(1, n + 1)}
    rid = {v: v for v in range(1, n + 1)}
    pairs = [(d[0][i] + d[0][j] - d[i][j], i, j)
             for i in range(1, n + 1) for j in range(i + 1, n + 1)]
    pairs.sort(reverse=True)
    for s, i, j in pairs:
        if s <= 0:
            break
        a, b = rid[i], rid[j]
        if a == b or load[a] + load[b] > cap:
            continue
        ra, rb = routes[a], routes[b]
        if ra[-1] == i and rb[0] == j:
            new = ra + rb
        elif ra[0] == i and rb[-1] == j:
            new = rb + ra
        elif ra[0] == i and rb[0] == j:
            new = ra[::-1] + rb
        elif ra[-1] == i and rb[-1] == j:
            new = ra + rb[::-1]
        else:
            continue
        routes[a] = new
        load[a] += load[b]
        del routes[b], load[b]
        for v in new:
            rid[v] = a
    return list(routes.values())


def local_search(d, dem, cap, routes, deadline, near):
    routes = [r for r in routes if r]
    loads = [sum(dem[v] for v in r) for r in routes]
    while time.perf_counter() < deadline:
        rid, pos = {}, {}
        for b, r in enumerate(routes):
            for k, v in enumerate(r):
                rid[v] = b
                pos[v] = k
        if (_relocate(d, dem, cap, routes, loads, rid, pos, near)
                or _swap(d, dem, cap, routes, loads, rid, pos, near)
                or _twoopt_star(d, dem, cap, routes, loads, rid, pos, near)
                or _twoopt(d, routes)):
            continue
        break
    return [r for r in routes if r]


def _relocate(d, dem, cap, routes, loads, rid, pos, near):
    for a, ra in enumerate(routes):
        for L in (1, 2, 3):
            for i in range(len(ra) - L + 1):
                seg = ra[i:i + L]
                s0, s1 = seg[0], seg[-1]
                sd = sum(dem[v] for v in seg)
                p = ra[i - 1] if i else 0
                q = ra[i + L] if i + L < len(ra) else 0
                gain = d[p][s0] + d[s1][q] - d[p][q]
                if gain <= 0:
                    continue
                rest = None
                for end in ((s0,) if L == 1 else (s0, s1)):
                    for u in near[end]:
                        b = rid[u]
                        if b == a:
                            if i <= pos[u] < i + L:
                                continue
                            if rest is None:
                                rest = ra[:i] + ra[i + L:]
                            t = rest
                            k = t.index(u)
                        else:
                            if loads[b] + sd > cap:
                                continue
                            t = routes[b]
                            k = pos[u]
                        for j in (k, k + 1):
                            x = t[j - 1] if j else 0
                            y = t[j] if j < len(t) else 0
                            c1 = d[x][s0] + d[s1][y] - d[x][y]
                            c2 = d[x][s1] + d[s0][y] - d[x][y]
                            if c1 < gain or c2 < gain:
                                s2 = seg if c1 <= c2 else seg[::-1]
                                nt = t[:j] + s2 + t[j:]
                                if b == a:
                                    routes[a] = nt
                                else:
                                    routes[a] = ra[:i] + ra[i + L:]
                                    routes[b] = nt
                                    loads[a] -= sd
                                    loads[b] += sd
                                return True
    return False


def _swap(d, dem, cap, routes, loads, rid, pos, near):
    for a, ra in enumerate(routes):
        for i, v in enumerate(ra):
            p = ra[i - 1] if i else 0
            q = ra[i + 1] if i + 1 < len(ra) else 0
            for w in near[v]:
                b = rid[w]
                if b == a:
                    continue
                rb = routes[b]
                j = pos[w]
                nl = loads[a] - dem[v] + dem[w]
                ml = loads[b] - dem[w] + dem[v]
                if nl > cap or ml > cap:
                    continue
                x = rb[j - 1] if j else 0
                y = rb[j + 1] if j + 1 < len(rb) else 0
                old = d[p][v] + d[v][q] + d[x][w] + d[w][y]
                new = d[p][w] + d[w][q] + d[x][v] + d[v][y]
                if new < old:
                    ra[i], rb[j] = w, v
                    loads[a], loads[b] = nl, ml
                    return True
    return False


def _twoopt_star(d, dem, cap, routes, loads, rid, pos, near):
    pref = []
    for r in routes:
        pf = [0]
        for v in r:
            pf.append(pf[-1] + dem[v])
        pref.append(pf)
    for a, ra in enumerate(routes):
        for i, u in enumerate(ra):
            an = ra[i + 1] if i + 1 < len(ra) else 0
            for w in near[u]:
                b = rid[w]
                if b == a:
                    continue
                rb = routes[b]
                j = pos[w]
                bp = rb[j - 1] if j else 0
                if d[u][w] + d[bp][an] < d[u][an] + d[bp][w]:
                    la = pref[a][i + 1] + loads[b] - pref[b][j]
                    lb = pref[b][j] + loads[a] - pref[a][i + 1]
                    if la <= cap and lb <= cap:
                        routes[a], routes[b] = ra[:i + 1] + rb[j:], rb[:j] + ra[i + 1:]
                        loads[a], loads[b] = la, lb
                        return True
    return False


def _twoopt(d, routes):
    for r in routes:
        m = len(r)
        for i in range(m - 1):
            a = r[i - 1] if i else 0
            for j in range(i + 1, m):
                b = r[j + 1] if j + 1 < m else 0
                if d[a][r[j]] + d[r[i]][b] < d[a][r[i]] + d[r[j]][b]:
                    r[i:j + 1] = r[i:j + 1][::-1]
                    return True
    return False


def perturb(d, instance, routes, rng, k):
    """Remove k related villages, reinsert by cheapest insertion. None if infeasible."""
    dem, cap, fleet, n = instance.demand, instance.capacity, instance.fleet, instance.size
    seed = rng.randint(1, n)
    near = sorted(range(1, n + 1), key=lambda v: d[seed][v] + rng.random() * 30)[:k]
    gone = set(near)
    rs = [[v for v in r if v not in gone] for r in routes]
    loads = [sum(dem[v] for v in r) for r in rs]
    rng.shuffle(near)
    for v in near:
        best = None
        for b, t in enumerate(rs):
            if loads[b] + dem[v] > cap:
                continue
            for j in range(len(t) + 1):
                x = t[j - 1] if j else 0
                y = t[j] if j < len(t) else 0
                c = d[x][v] + d[v][y] - d[x][y]
                if best is None or c < best[0]:
                    best = (c, b, j)
        if best is None:
            if sum(1 for r in rs if r) >= fleet:
                return None
            rs.append([v])
            loads.append(dem[v])
        else:
            _, b, j = best
            rs[b].insert(j, v)
            loads[b] += dem[v]
    return [r for r in rs if r]


class MySolver(Solver):
    def solve(self, instance, submit_candidate):
        d = distance_matrix(instance)
        dem, cap, fleet = instance.demand, instance.capacity, instance.fleet
        rng = random.Random(12345)
        K = min(14, instance.size - 1)
        near = [[]] + [sorted((u for u in range(1, instance.size + 1) if u != v), key=lambda u: d[v][u])[:K]
                       for v in range(1, instance.size + 1)]

        starter = StarterSolver().solve(instance, submit_candidate)["routes"]
        starter = [list(r) for r in starter if r]
        receipt = submit_candidate({"routes": starter})
        deadline = time.perf_counter() + receipt["remaining_s"] - SAFETY_S

        best, best_c = starter, cost(d, starter)
        cands = [starter]
        sv = savings(d, instance)
        if len(sv) <= fleet:
            cands.append(sv)
        for c in cands:
            r = local_search(d, dem, cap, [list(x) for x in c], deadline, near)
            rc = cost(d, r)
            if rc < best_c and len(r) <= fleet:
                best, best_c = r, rc
                submit_candidate({"routes": [list(x) for x in best]})

        cur, cur_c = best, best_c
        n = instance.size
        kmax = max(3, min(15, n // 6))
        T0 = best_c / max(1, n) * 0.15
        t_start = time.perf_counter()
        while time.perf_counter() < deadline:
            frac = (time.perf_counter() - t_start) / max(1e-6, deadline - t_start)
            temp = T0 * (1 - frac)
            cand = perturb(d, instance, cur, rng, rng.randint(2, kmax))
            if cand is None:
                continue
            cand = local_search(d, dem, cap, cand, deadline, near)
            if len(cand) > fleet:
                continue
            c = cost(d, cand)
            if c <= cur_c or (temp > 0 and rng.random() < 2.718 ** (-(c - cur_c) / temp)):
                cur, cur_c = cand, c
                if c < best_c:
                    best, best_c = cand, c
                    submit_candidate({"routes": [list(x) for x in best]})
        return {"routes": [list(x) for x in best]}
