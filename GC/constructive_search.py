from __future__ import annotations
from roar_net_api.operations import (
    SupportsApplyMove,
    SupportsMoves,
    SupportsLowerBoundIncrement,
)
from typing import Iterable, final, TYPE_CHECKING

if TYPE_CHECKING:
    from solver import Solution


@final
class AddMove(SupportsApplyMove["Solution"], SupportsLowerBoundIncrement["Solution"]):
    def __init__(self, neighbourhood, n: int, c: int):
        self.neighbourhood = neighbourhood
        # n is node
        self.n = n
        self.c = c

    def apply_move(self, solution: "Solution") -> "Solution":
        # Update lower bound
        if self.c >= solution.used_colors:
            solution.used_colors += 1
            solution.lb = max(solution.lb, solution.used_colors)
        solution.colors[self.n] = self.c
        solution.color_map[self.c].append(self.n)
        solution.not_colored.remove(self.n)
        return solution

    def lower_bound_increment(self, solution: "Solution") -> float:
        return max(0, self.c - solution.lb)


# ------------------------------- Neighbourhood ------------------------------


@final
class AddNeighbourhood(SupportsMoves["Solution", AddMove]):
    def __init__(self, problem):
        self.problem = problem

    def moves(self, solution: "Solution") -> Iterable[AddMove]:
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
