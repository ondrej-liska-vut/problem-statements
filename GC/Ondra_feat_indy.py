from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Protocol, TypeVar
import random
import time

# ---- ROAR-NET operations (import nebo generický fallback pro Py 3.13) ----
try:
    from roar_net_api.operations import (
        SupportsEmptySolution,
        SupportsConstructiveNeighbourhood,
        SupportsApplyMove,
    )
except Exception:
    S = TypeVar("S")
    class SupportsApplyMove(Protocol[S]):
        def apply_move(self, solution: S) -> S: ...
    class SupportsEmptySolution(Protocol[S]):
        def empty_solution(self) -> S: ...
    class SupportsConstructiveNeighbourhood(Protocol):
        def constructive_neighbourhood(self): ...

# ----------------- Pomocné -----------------
def mex(used: set[int]) -> int:
    m = 0
    while m in used:
        m += 1
    return m

# ----------------- Moves -----------------
@dataclass(frozen=True)
class AssignMove(SupportsApplyMove["GCSolution"]):
    v: int
    c: int
    def apply_move(self, solution: "GCSolution") -> "GCSolution":
        colors = list(solution.colors)
        colors[self.v] = self.c
        return GCSolution(solution.problem, tuple(colors))

@dataclass(frozen=True)
class RecolorMove(SupportsApplyMove["GCSolution"]):
    v: int
    c: int
    def apply_move(self, solution: "GCSolution") -> "GCSolution":
        if solution.colors[self.v] == self.c:
            return solution
        colors = list(solution.colors)
        colors[self.v] = self.c
        return GCSolution(solution.problem, tuple(colors))

# ----------------- Neighbourhoods -----------------
class GCConstructiveNeighbourhood:
    def __init__(self, problem: "GCProblem"):
        self.problem = problem
    def moves(self, s: "GCSolution") -> Iterable[AssignMove]:
        try:
            v = next(i for i, col in enumerate(s.colors) if col is None)
        except StopIteration:
            return []
        used = {c for c in s.colors if c is not None}
        opts = sorted(used) + [mex(used)]
        return (AssignMove(v, c) for c in opts)

class GCLocalNeighbourhood:
    def __init__(self, problem: "GCProblem"):
        self.problem = problem
    def moves(self, s: "GCSolution") -> Iterable[RecolorMove]:
        used = {c for c in s.colors if c is not None}
        newc = mex(used)
        palette = sorted(used) + [newc]
        for v in range(self.problem.n):
            cv = s.colors[v]
            for c in palette:
                if c != cv:
                    yield RecolorMove(v, c)

# ----------------- Solution -----------------
@dataclass(frozen=True)
class GCSolution:
    problem: "GCProblem"
    colors: Tuple[Optional[int], ...]  # None = neobarveno

    def used_colors(self) -> int:
        return len({c for c in self.colors if c is not None})

    def _conflicts(self) -> int:
        cols = self.colors
        conflicts = 0
        for u, v in self.problem.edges:
            cu, cv = cols[u], cols[v]
            if cu is not None and cv is not None and cu == cv:
                conflicts += 1
        return conflicts

    def objective_value(self) -> float:
        conflicts = self._conflicts()
        incomplete = any(c is None for c in self.colors)
        feasible = (not incomplete) and (conflicts == 0)
        val = float(self.used_colors())
        if not feasible:
            val += 50.0
        if self.used_colors() > self.problem.best_used_colors:
            val += 20.0
        val += 0.001 * conflicts  # jemný tie-breaker
        return val

    def is_complete(self) -> bool:
        return all(c is not None for c in self.colors)

    def is_feasible(self) -> bool:
        if not self.is_complete():
            return False
        return self._conflicts() == 0

# ----------------- Problem -----------------
class GCProblem(SupportsEmptySolution["GCSolution"], SupportsConstructiveNeighbourhood):
    def __init__(self, n: int, edges: List[Tuple[int, int]]):
        self.n = n
        self.edges = sorted(set((min(u, v), max(u, v)) for u, v in edges if u != v))
        adj = [[] for _ in range(n)]
        for u, v in self.edges:
            adj[u].append(v); adj[v].append(u)
        self.adj = tuple(tuple(ns) for ns in adj)
        self.best_used_colors = n + 1  # aktualizuje se při feasible řešeních

    def empty_solution(self) -> GCSolution:
        return GCSolution(self, tuple([None] * self.n))

    def constructive_neighbourhood(self) -> GCConstructiveNeighbourhood:
        return GCConstructiveNeighbourhood(self)

    def local_neighbourhood(self) -> GCLocalNeighbourhood:
        return GCLocalNeighbourhood(self)

# ----------------- Heuristiky -----------------
def _update_best_used_colors(problem: GCProblem, s: GCSolution) -> None:
    if s.is_feasible():
        u = s.used_colors()
        if u < problem.best_used_colors:
            problem.best_used_colors = u

def greedy_construct(problem: GCProblem, rng: random.Random) -> GCSolution:
    s = problem.empty_solution()
    neigh = problem.constructive_neighbourhood()
    while not s.is_complete():
        cand = [(mv.apply_move(s).objective_value(), mv.apply_move(s)) for mv in neigh.moves(s)]
        best_val = min(val for val, _ in cand)
        bests = [s2 for val, s2 in cand if val == best_val]
        s = rng.choice(bests)
    _update_best_used_colors(problem, s)
    return s

def first_improvement(problem: GCProblem, s: GCSolution, rng: random.Random, max_iters: int = 100_000) -> GCSolution:
    neigh = problem.local_neighbourhood()
    cur = s
    cur_val = cur.objective_value()
    _update_best_used_colors(problem, cur)
    for _ in range(max_iters):
        moves = list(neigh.moves(cur))
        rng.shuffle(moves)
        improved = False
        for mv in moves:
            cand = mv.apply_move(cur)
            val = cand.objective_value()
            if val < cur_val - 1e-12:
                cur, cur_val = cand, val
                _update_best_used_colors(problem, cur)
                improved = True
                break
        if not improved:
            break
    _update_best_used_colors(problem, cur)
    return cur

# ----------------- Orchestrátor: 50 restartů -----------------
def solve_graph_coloring(n: int,
                         edges: List[Tuple[int, int]],
                         seed: int = 0,
                         restarts: int = 50,
                         time_limit_s: float | None = None) -> GCSolution:
    start = time.monotonic()
    global_best_used = float("inf")
    best: GCSolution | None = None

    for r in range(restarts):
        if time_limit_s is not None and (time.monotonic() - start) >= time_limit_s:
            break

        prob = GCProblem(n, edges)
        if global_best_used < float("inf"):
            prob.best_used_colors = int(global_best_used)
        rng = random.Random(seed + r)

        s0 = greedy_construct(prob, rng)
        s1 = first_improvement(prob, s0, rng)

        if best is None or s1.objective_value() < best.objective_value():
            best = s1
        if s1.is_feasible():
            u = s1.used_colors()
            if u < global_best_used:
                global_best_used = u

    return best if best is not None else GCProblem(n, edges).empty_solution()

# ----------------- Generátory příkladů -----------------
def mycielski_graph(t: int):
    if t < 2:
        raise ValueError("t >= 2")
    n = 2
    E = {(0, 1)}
    for _ in range(3, t + 1):
        newE = set(E)
        for (i, j) in E:
            newE.add((min(i, n + j), max(i, n + j)))
            newE.add((min(j, n + i), max(j, n + i)))
        w = 2 * n
        for i in range(n):
            newE.add((min(w, n + i), max(w, n + i)))
        E = newE
        n = 2 * n + 1
    edges = sorted((min(u, v), max(u, v)) for (u, v) in E)
    return n, edges

def queen_graph(m: int, n: int):
    edges = set()
    for r1 in range(m):
        for c1 in range(n):
            v1 = r1 * n + c1
            for r2 in range(m):
                for c2 in range(n):
                    v2 = r2 * n + c2
                    if v2 <= v1:
                        continue
                    if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                        edges.add((v1, v2))
    return m * n, sorted(edges)

# ----------------- Použití -----------------
if __name__ == "__main__":
    # 1) Malý příklad
    n = 5
    edges = [(0,1),(1,2),(2,3),(3,4),(4,0),(0,2)]
    sol = solve_graph_coloring(n, edges, seed=42, restarts=50)
    print("Malý graf:")
    print("  used_colors:", sol.used_colors())
    print("  feasible:", sol.is_feasible())
    print("  objective:", sol.objective_value())
    print("  colors:", sol.colors)

    # 2) Mycielski M5 (těžší)
    n2, e2 = mycielski_graph(5)
    sol2 = solve_graph_coloring(n2, e2, seed=1, restarts=50)
    print("\nMycielski M5:")
    print("  used_colors:", sol2.used_colors())
    print("  feasible:", sol2.is_feasible())
    print("  objective:", sol2.objective_value())

    # 3) Queen 8x8
    n3, e3 = queen_graph(8, 8)
    sol3 = solve_graph_coloring(n3, e3, seed=2, restarts=50)
    print("\nQueen 8x8:")
    print("  used_colors:", sol3.used_colors())
    print("  feasible:", sol3.is_feasible())
    print("  objective:", sol3.objective_value())
