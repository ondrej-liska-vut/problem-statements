import networkx
from dataclasses import dataclass
from typing import Optional, Iterable, Protocol, Self, TextIO, TypeVar, final

from roar_net_api.operations import (
    SupportsApplyMove,
    SupportsEmptySolution,
    SupportsConstructionNeighbourhood,
    SupportsMoves,
    SupportsLowerBoundIncrement,
)


# --- Solution ---
@final
class Solution():
    def __init__(self, problem, colors: list[Optional[int]], lb: int):
        self.problem = problem
        self.colors = colors
        self.not_colored = [i for i, c in enumerate(colors) if c is None]
        self.lb = lb
        self.used_colors = len({c for c in self.colors if c is not None}) 

    # def used_colors(self) -> int:
    #     return len({c for c in self.colors if c is not None})

    def to_textio(self) -> None:
        pass

    # def conflicts(self) -> int:
    #     cols = self.colors
    #     cnt = 0
    #     for u, v in self.problem.edges:
    #         cu, cv = cols[u], cols[v]
    #         if cu is not None and cv is not None and cu == cv:
    #             cnt += 1
    #     return cnt
    
    def objective_value(self) -> Optional[int]:
        if self.is_feasible:
            return self.lb
        return None

    def is_complete(self) -> bool:
        return all(c is not None for c in self.colors)

    @property
    def is_feasible(self) -> bool:
        return self.is_complete() # and self.conflicts() == 0
    
# ----------------------------------- Moves -----------------------------------


@final
class AddMove(SupportsApplyMove[Solution], SupportsLowerBoundIncrement[Solution]):
    def __init__(self, neighbourhood, n: int, c: int):
        self.neighbourhood = neighbourhood
        # n is node
        self.n = n
        self.c = c

    def apply_move(self, solution: Solution) -> Solution:
        # Update lower bound
        if self.c > solution.used_colors:
            solution.used_colors += 1
            solution.lb = max(solution.lb, solution.used_colors)
        solution.colors[self.n] = self.c
        solution.not_colored.remove(self.n)
        return solution

    def lower_bound_increment(self, solution: Solution) -> float:
        return max(0, self.c - solution.lb)


# ------------------------------- Neighbourhood ------------------------------


@final
class AddNeighbourhood(SupportsMoves[Solution, AddMove]):
    def __init__(self, problem):
        self.problem = problem

    def moves(self, solution: Solution) -> Iterable[AddMove]:
        assert self.problem == solution.problem
        
        for n in solution.not_colored: #for non colored nodes
            neigbouring_colors = [solution.colors[j] for j in self.problem.g.neighbors(n)]
            used_colors = set(color for color in neigbouring_colors if color)
            available_colors = [color for color in range(solution.used_colors + 1) if color not in used_colors]
            for c in available_colors:
                yield AddMove(self, n, c)


# ---------------------------------- Problem --------------------------------

@final
class Problem(
    SupportsConstructionNeighbourhood[AddNeighbourhood],
    SupportsEmptySolution[Solution],
):
    def __init__(self, g: networkx.Graph, name: str):
        self.name = name
        self.c_nbhood: Optional[AddNeighbourhood] = None
        # self.l_nbhood: Optional[TwoOptNeighbourhood] = None
        self.g = g

    # def __str__(self) -> str:
    #     out: list[str] = []
    #     for row in self.dist:
    #         out.append(" ".join(map(str, row)))
    #     return "\n".join(out)

    def construction_neighbourhood(self) -> AddNeighbourhood:
        if self.c_nbhood is None:
            self.c_nbhood = AddNeighbourhood(self)
        return self.c_nbhood

    # def local_neighbourhood(self) -> TwoOptNeighbourhood:
    #     if self.l_nbhood is None:
    #         self.l_nbhood = TwoOptNeighbourhood(self)
    #     return self.l_nbhood

    def empty_solution(self) -> Solution:
        return Solution(self, [None] * len(self.g), 0) # TODO better initial lb

    # def random_solution(self) -> Solution:
    #     c = list(range(1, self.n))
    #     random.shuffle(c)
    #     c.insert(0, 0)
    #     obj = self.dist[c[-1]][c[0]]
    #     for ix in range(1, self.n):
    #         obj += self.dist[c[ix - 1]][c[ix]]
    #     return Solution(self, c, set(), obj)

if __name__ == "__main__":
    import roar_net_api.algorithms as alg
    from parser import IOParser

    G = IOParser.parse2nx("problems/graph-coloring/data/0-SmallExample/0-SmallExample.col")
    problem = Problem(G, "0-SmallExample")

    # Run greedy construction to get an initial solution
    solution = alg.greedy_construction(problem)
    # solution = alg.beam_search(problem, bw=10)
    # solution = alg.grasp(problem, 30.0)

    # Run simulated annealing to improve the previous solution
    # solution = alg.sa(problem, solution, 10.0, 30.0)
    # solution = alg.rls(problem, solution, 10.0)
    # solution = alg.best_improvement(problem, solution)
    # solution = alg.first_improvement(problem, solution)

    # Print the final solution to stdout
    # solution.to_textio(sys.stdout)