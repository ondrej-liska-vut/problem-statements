from __future__ import annotations
import random
from typing import Iterable, final, TYPE_CHECKING
from roar_net_api.operations import (
    SupportsApplyMove,
    SupportsRandomMovesWithoutReplacement,
    SupportsObjectiveValueIncrement,
)

if TYPE_CHECKING:
    from solver import Solution


@final
class OneRecolorMove(
    SupportsApplyMove["Solution"], SupportsObjectiveValueIncrement["Solution"]
):
    """A move that recolors a single node for local search only."""

    def __init__(self, neighbourhood: OneRecolorNeighbourhood, n: int, c: int):
        """A move that recolors a single node for local search only.

        Args:
            neighbourhood (OneRecolorNeighbourhood): The neighbourhood this move belongs to.
            n (int): The node to recolor.
            c (int): The new color for the node.
        """
        self.neighbourhood = neighbourhood
        self.n = n
        self.c = c

    def apply_move(self, solution: "Solution") -> "Solution":
        """Applies the recolor move to the given solution.

        Args:
            solution (Solution): The solution to apply the move to.

        Returns:
            Solution: The modified solution after applying the move.
        """
        old_color = solution.colors[self.n]
        new_color = self.c
        obj_increment = self.objective_value_increment(solution)

        solution.colors[self.n] = new_color
        if old_color is None:
            solution.not_colored.remove(self.n)
        else:
            solution.color_map[old_color].remove(self.n)
            solution.color_map[new_color].append(self.n)
            if not solution.color_map[old_color]:
                del solution.color_map[old_color]
            solution.used_colors = len(solution.color_map)

        solution.lb += obj_increment
        solution.update_objective_value(obj_increment + solution.objective_value())
        return solution

    def objective_value_increment(self, solution: "Solution") -> float:
        """Calculates the objective value increment for the move.

        Args:
            solution (Solution): The solution to evaluate.

        Returns:
            float: The objective value increment.
        """
        colors_around = solution.colors_around(self.n)
        conflict_after = colors_around.count(self.c)
        conflict_before = colors_around.count(solution.colors[self.n])
        if self.c not in solution.color_map.keys():
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
        ) * solution.problem.conflict_penalty + num_colors_increment


@final
class OneRecolorNeighbourhood(
    SupportsRandomMovesWithoutReplacement["Solution", OneRecolorMove],
):
    """A neighbourhood for one recolor moves."""

    def __init__(self, problem):
        self.problem = problem

    def random_moves_without_replacement(
        self, solution: "Solution"
    ) -> Iterable[OneRecolorMove]:
        """Generates random one recolor moves without replacement.

        Args:
            solution (Solution): The solution to generate moves for.

        Yields:
            Iterator[Iterable[OneRecolorMove]]: An iterator over the generated moves.
        """
        assert self.problem == solution.problem
        N = len(solution.colors)
        randomized_nodes = list(range(N))
        random.shuffle(randomized_nodes)
        for n in randomized_nodes:
            randomized_colors = list(range(0, solution.used_colors + 1))
            random.shuffle(randomized_colors)
            for c in randomized_colors:
                yield OneRecolorMove(self, n, c)
