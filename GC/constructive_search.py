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
    """A constructive move that adds a color to a node.

    Args:
        SupportsApplyMove (SupportsApplyMove): The base class for applying moves.
        SupportsLowerBoundIncrement (SupportsLowerBoundIncrement): The base class for lower bound increments.
    """

    def __init__(self, neighbourhood: AddNeighbourhood, n: int, c: int):
        """Initializes the AddMove.

        Args:
            neighbourhood (AddNeighbourhood): The neighbourhood this move belongs to.
            n (int): The node to add a color to.
            c (int): The color to add.
        """
        self.neighbourhood = neighbourhood
        self.n = n
        self.c = c

    def apply_move(self, solution: "Solution") -> "Solution":
        """Applies the add move to the given solution.

        Args:
            solution (Solution): The solution to apply the move to.

        Returns:
            Solution: The modified solution after applying the move.
        """
        if self.c >= solution.used_colors:
            solution.used_colors += 1
            solution.lb = max(solution.lb, solution.used_colors)
        solution.colors[self.n] = self.c
        solution.color_map[self.c].append(self.n)
        solution.not_colored.remove(self.n)
        # solution.nodes_available_colors = solution.nodes_available_colors_method()
        for neighbor in solution.problem.g.neighbors(self.n):
            if self.c in solution.nodes_available_colors[neighbor]:
                solution.nodes_available_colors[neighbor].remove(self.c)
        return solution

    def lower_bound_increment(self, solution: "Solution") -> float:
        """Calculates the lower bound increment for the move.

        Args:
            solution (Solution): The solution to evaluate.

        Returns:
            float: The lower bound increment.
        """
        return max(0, self.c - solution.lb)


@final
class AddNeighbourhood(SupportsMoves["Solution", AddMove]):
    """A neighbourhood for add moves.

    Args:
        SupportsMoves (SupportsMoves): The base class for move neighbourhoods.
    """

    def __init__(self, problem):
        """Initializes the AddNeighbourhood.

        Args:
            problem (Problem): The problem this neighbourhood belongs to.
        """
        self.problem = problem

    def moves(self, solution: "Solution") -> Iterable[AddMove]:
        """Generates all possible add moves for the given solution.

        Args:
            solution (Solution): The solution to generate moves for.

        Returns:
            Iterable[AddMove]: An iterable of all possible add moves.
        """
        assert self.problem == solution.problem

        for n in solution.not_colored:  # for non-colored nodes
            max_color = solution.used_colors
            available_colors = [
                color
                for color in solution.nodes_available_colors[n]
                if color <= max_color
            ]

            for c in available_colors:
                yield AddMove(self, n, c)
