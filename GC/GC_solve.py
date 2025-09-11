import networkx
from dataclasses import dataclass
from typing import Optional, Iterable, Protocol, Self, TextIO, TypeVar, final
import random
from collections import defaultdict

from roar_net_api.operations import (
    SupportsApplyMove,
    SupportsEmptySolution,
    SupportsConstructionNeighbourhood,
    SupportsMoves,
    SupportsLowerBoundIncrement,
    SupportsLocalNeighbourhood,
    SupportsRandomMovesWithoutReplacement,
    SupportsObjectiveValueIncrement,
)


# --- Solution ---
@final
class Solution:
    def __init__(self, problem, colors: list[Optional[int]], lb: float):
        self.problem = problem
        self.colors = colors
        self.not_colored = [i for i, c in enumerate(colors) if c is None]
        self.lb = lb
        self.used_colors = len({c for c in self.colors if c is not None})
        self.color_map = defaultdict(list)
        for n, c in enumerate(colors):
            if c is not None:
                self.color_map[c].append(n)

    # def used_colors(self) -> int:
    #     return len({c for c in self.colors if c is not None})

    def to_textio(self) -> None:
        pass

    def conflicts(self) -> int:
        cols = self.colors
        cnt = 0
        for u, v in self.problem.edges:
            cu, cv = cols[u], cols[v]
            if cu is not None and cv is not None and cu == cv:
                cnt += 1
        return cnt

    def objective_value(self) -> Optional[int]:
        if self.is_feasible:
            return self.used_colors
        return self.used_colors + self.problem.inf_penalty

    def is_complete(self) -> bool:
        # return all(c is not None for c in self.colors)
        return self.not_colored == []

    @property
    def is_feasible(self) -> bool:
        return self.is_complete() and self.conflicts() == 0

    def colors_around(self, noode: int) -> list[int]:
        return [
            self.colors[neigh]
            for neigh in self.problem.g.neighbors(noode)
            if self.colors[neigh] is not None
        ]


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
        solution.color_map[self.c].append(self.n)
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

        for n in solution.not_colored:  # for non colored nodes
            neigbouring_colors = [
                solution.colors[j] for j in self.problem.g.neighbors(n)
            ]
            used_colors = set(
                color for color in neigbouring_colors if color is not None
            )
            available_colors = [
                color
                for color in range(solution.used_colors + 2)
                if color not in used_colors
            ]
            for c in available_colors:
                yield AddMove(self, n, c)


@final
class OneRecolorMove(
    SupportsApplyMove[Solution], SupportsObjectiveValueIncrement[Solution]
):
    def __init__(self, neighbourhood, n: int, c: int):
        self.neighbourhood = neighbourhood
        # ix and jx are indices
        self.n = n
        self.c = c

    def apply_move(self, solution: Solution) -> Solution:
        old_color = solution.colors[self.n]
        new_color = self.c

        solution.colors[self.n] = new_color
        if old_color is None:
            solution.not_colored.remove(self.n)
        else:
            solution.color_map[old_color].remove(self.n)
            solution.color_map[new_color].append(self.n)
            if not solution.color_map[old_color]:
                del solution.color_map[old_color]
            solution.used_colors = len(solution.color_map)
        solution.lb += self.objective_value_increment(solution)
        return solution

    def objective_value_increment(self, solution: Solution) -> float:
        colors_around = solution.colors_around(self.n)
        conf_neb_before_move = colors_around + [solution.colors[self.n]]
        conf_neb_after_move = colors_around + [self.c]

        for c in set(colors_around):
            conf_neb_before_move.remove(c)
            conf_neb_after_move.remove(c)

        conflict_before = (
            len(conf_neb_before_move) - 1
        )  # -1 because we count the node itself
        conflict_after = (
            len(conf_neb_after_move) - 1
        )  # -1 because we count the node itself

        if self.c not in solution.color_map:
            num_colors_increment = 1
        elif (
            len(solution.color_map[solution.colors[self.n]]) == 1
            and solution.colors[self.n] != self.c
        ):
            num_colors_increment = -1
        else:
            num_colors_increment = 0

        return (
            conflict_after - conflict_before
        ) * solution.problem.inf_penalty + num_colors_increment


@final
class OneRecolorNeighbourhood(
    SupportsRandomMovesWithoutReplacement[Solution, OneRecolorMove],
):
    def __init__(self, problem):
        self.problem = problem

    def random_moves_without_replacement(
        self, solution: Solution
    ) -> Iterable[OneRecolorMove]:
        assert self.problem == solution.problem
        N = len(solution.colors)
        # This is only meant to be used as a local neighbourhood, so solution should be feasible
        assert solution.is_feasible

        for n in random.shuffle(list(range(1, N + 1))):
            for c in random.shuffle(list(range(1, solution.used_colors + 2))):
                yield OneRecolorMove(self, n)


# ---------------------------------- Problem --------------------------------


@final
class Problem(
    SupportsConstructionNeighbourhood[AddNeighbourhood],
    SupportsEmptySolution[Solution],
    SupportsLocalNeighbourhood[OneRecolorNeighbourhood],
):
    def __init__(self, G: networkx.Graph, name: str, inf_penalty=1000):
        self.name = name
        self.c_nbhood: Optional[AddNeighbourhood] = None
        # self.l_nbhood: Optional[TwoOptNeighbourhood] = None
        mapping = {old: old - 1 for old in G.nodes()}
        G_relabelled = networkx.relabel_nodes(G, mapping)
        self.g = G_relabelled
        self.inf_penalty = inf_penalty

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
        return Solution(self, [None] * len(self.g), 0)  # TODO better initial lb

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

    name = "1-FullIns_4"
    G = IOParser.parse2nx(f"problems/graph-coloring/data/{name}/{name}.col")
    problem = Problem(G, name)

    # Run greedy construction to get an initial solution
    solution = alg.greedy_construction(problem)
    # solution = alg.beam_search(problem, bw=10)
    # solution = alg.grasp(problem, 30.0)

    # Run simulated annealing to improve the previous solution
    # solution = alg.sa(problem, solution, 10.0, 30.0)
    # solution = alg.rls(problem, solution, 10.0)
    # solution = alg.best_improvement(problem, solution)
    # solution = alg.first_improvement(problem, solution)

    print(solution.colors)
    # Print the final solution to stdout
    # solution.to_textio(sys.stdout)
