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


def local_search(d, dem, cap, routes, deadline):
    loads = [sum(dem[v] for v in r) for r in routes]
    improved = True
    while improved:
        improved = False
        if time.perf_counter() > deadline:
            break
        # --- segment relocate (len 1..3, both orientations), inter + intra
        for a in range(len(routes)):
            ra = routes[a]
            for L in (1, 2, 3):
                for i in range(len(ra) - L + 1):
                    seg = ra[i:i + L]
                    sl = sum(dem[v] for v in seg)
                    p = ra[i - 1] if i else 0
                    q = ra[i + L] if i + L < len(ra) else 0
                    gain = d[p][seg[0]] + d[seg[-1]][q] - d[p][q]
                    rest = ra[:i] + ra[i + L:]
                    for b in range(len(routes)):
                        if b == a:
                            t = rest
                        else:
                            if loads[b] + sl > cap:
                                continue
                            t = routes[b]
                        for j in range(len(t) + 1):
                            x = t[j - 1] if j else 0
                            y = t[j] if j < len(t) else 0
                            c1 = d[x][seg[0]] + d[seg[-1]][y] - d[x][y]
                            c2 = d[x][seg[-1]] + d[seg[0]][y] - d[x][y]
                            if c1 < gain or c2 < gain:
                                s2 = seg if c1 <= c2 else seg[::-1]
                                nt = t[:j] + s2 + t[j:]
                                if b == a:
                                    routes[a] = nt
                                else:
                                    routes[a] = rest
                                    routes[b] = nt
                                    loads[a] -= sl
                                    loads[b] += sl
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # --- swap one village between routes
        for a in range(len(routes)):
            ra = routes[a]
            for b in range(a + 1, len(routes)):
                rb = routes[b]
                for i, v in enumerate(ra):
                    p = ra[i - 1] if i else 0
                    q = ra[i + 1] if i + 1 < len(ra) else 0
                    for j, w in enumerate(rb):
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
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # --- 2-opt* between routes (swap tails)
        for a in range(len(routes)):
            ra = routes[a]
            pa = [0]
            for v in ra:
                pa.append(pa[-1] + dem[v])
            for b in range(a + 1, len(routes)):
                rb = routes[b]
                pb = [0]
                for v in rb:
                    pb.append(pb[-1] + dem[v])
                for i in range(len(ra) + 1):
                    ap = ra[i - 1] if i else 0
                    an = ra[i] if i < len(ra) else 0
                    for j in range(len(rb) + 1):
                        bp = rb[j - 1] if j else 0
                        bn = rb[j] if j < len(rb) else 0
                        if pa[i] + pb[-1] - pb[j] > cap or pb[j] + pa[-1] - pa[i] > cap:
                            continue
                        if d[ap][bn] + d[bp][an] < d[ap][an] + d[bp][bn]:
                            routes[a], routes[b] = ra[:i] + rb[j:], rb[:j] + ra[i:]
                            loads[a] = pa[i] + pb[-1] - pb[j]
                            loads[b] = pb[j] + pa[-1] - pa[i]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # --- 2-opt inside routes
        for r in routes:
            m = len(r)
            for i in range(m - 1):
                a = r[i - 1] if i else 0
                for j in range(i + 1, m):
                    b = r[j + 1] if j + 1 < m else 0
                    if d[a][r[j]] + d[r[i]][b] < d[a][r[i]] + d[r[j]][b]:
                        r[i:j + 1] = r[i:j + 1][::-1]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return [r for r in routes if r]


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
            r = local_search(d, dem, cap, [list(x) for x in c], deadline)
            rc = cost(d, r)
            if rc < best_c and len(r) <= fleet:
                best, best_c = r, rc
                submit_candidate({"routes": [list(x) for x in best]})

        cur, cur_c = best, best_c
        n = instance.size
        kmax = max(3, min(15, n // 6))
        T0 = best_c / max(1, n) * 0.05
        t_start = time.perf_counter()
        while time.perf_counter() < deadline:
            frac = (time.perf_counter() - t_start) / max(1e-6, deadline - t_start)
            temp = T0 * (1 - frac)
            cand = perturb(d, instance, cur, rng, rng.randint(2, kmax))
            if cand is None:
                continue
            cand = local_search(d, dem, cap, cand, deadline)
            if len(cand) > fleet:
                continue
            c = cost(d, cand)
            if c <= cur_c or (temp > 0 and rng.random() < 2.718 ** (-(c - cur_c) / temp)):
                cur, cur_c = cand, c
                if c < best_c:
                    best, best_c = cand, c
                    submit_candidate({"routes": [list(x) for x in best]})
        return {"routes": [list(x) for x in best]}
