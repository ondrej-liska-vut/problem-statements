from __future__ import annotations
from copy import deepcopy
import networkx
import argparse
from typing import Optional, Self, final
from collections import defaultdict
from constructive_search import AddNeighbourhood
from local_search import OneRecolorNeighbourhood
import roar_net_api.algorithms as alg
from parser import IOParser
from roar_net_api.operations import (
    SupportsEmptySolution,
    SupportsConstructionNeighbourhood,
    SupportsLocalNeighbourhood,
    SupportsCopySolution,
    SupportsObjectiveValue,
)


@final
class Solution(SupportsCopySolution, SupportsObjectiveValue):
    """Represents a solution to the graph coloring problem.
    The solution consists of a list of colors assigned to each node,
    a lower bound on the number of colors used, and a reference to the problem instance.
    The objective value is calculated based on the number of conflicts and used colors.

    Args:
        SupportsCopySolution (SupportsCopySolution): The base class for copying solutions.
        SupportsObjectiveValue (SupportsObjectiveValue): The base class for objective value calculations.
    """

    def __init__(
        self,
        problem,
        colors: list[Optional[int]],
        lb: float,
        nodes_available_colors: Optional[list[list[int]]] = None,
    ):
        """Initializes the solution.

        Args:
            problem (_type_): _description_
            colors (list[Optional[int]]): A list of colors assigned to each node.
            lb (float): A lower bound on the number of colors used.
            nodes_available_colors (Optional[list[list[int]]], optional): A list of available colors for each node.
            If none the value for each node is determined by the node's degree. Defaults to None.
        """
        self.problem = problem
        self.colors = colors
        self.not_colored = [i for i, c in enumerate(colors) if c is None]
        self.lb = lb
        self.used_colors = len({c for c in self.colors if c is not None})
        self.nodes_available_colors = (
            nodes_available_colors
            if nodes_available_colors is not None
            else self._nodes_available_colors()
        )
        self.color_map = defaultdict(list)
        self._objective_value = None
        for n, c in enumerate(colors):
            if c is not None:
                self.color_map[c].append(n)

    def nodes_max_available_colors(self) -> list[int]:
        """Returns a list of maximum available colors for each node.

        Returns:
            list[int]: A list of maximum available colors for each node.
        """
        max_available_colors: list[int] = [0] * len(self.problem.g.nodes)
        for node in range(len(self.problem.g.nodes)):
            count = len(list(self.problem.g.neighbors(node))) + 1
            max_available_colors[node] = count
        return max_available_colors

    def _nodes_available_colors(self) -> list[list[int]]:
        """Returns a list of degree based available colors for each node.

        Returns:
            list[list[int]]: A list of available colors for each node.
        """
        max_colors_list = self.nodes_max_available_colors()
        available_colors = []
        for node, max_colors in enumerate(max_colors_list):
            # Possible colors: 0, 1, ..., max_count-1
            possible_colors = list(range(max_colors))
            # Remove colors already used in the neighborhood
            loc_used_colors = set(self.colors_around(node))
            node_available = [c for c in possible_colors if c not in loc_used_colors]
            available_colors.append(node_available)
        return available_colors

    def to_textio(self) -> str:
        """Converts the solution to a text representation in format <node_id> <color_id>.

        Returns:
            str: The text representation of the solution.
        """
        # print(" Colors of nodes:",self.colors)
        result = ""
        for i, c in enumerate(self.colors):
            result += str(i) + " " + str(c) + "\n"
        return result

    def conflicts(self) -> int:
        """Calculates the number of conflicts in the solution.

        Returns:
            int: The number of conflicts.
        """
        cols = self.colors
        cnt = 0
        for u, v in self.problem.g.edges:
            cu, cv = cols[u], cols[v]
            if cu is not None and cv is not None and cu == cv:
                cnt += 1
        return cnt

    def update_objective_value(self, value: Optional[float]) -> Optional[float]:
        """Updates the objective value of the solution."""
        self._objective_value = value
        return self._objective_value

    def objective_value(self) -> float:
        """Calculates the objective value of the solution. The value is calculated only
        if it is not already set to prevent unnecessary recomputation.

        Returns:
            float: The objective value of the solution.
        """
        if self._objective_value is not None:
            return self._objective_value
        else:
            self._objective_value = (
                self.conflicts() * self.problem.conflict_penalty + self.used_colors
            )
            return self._objective_value

    def is_complete(self) -> bool:
        """Checks if the solution is complete (all nodes are colored).

        Returns:
            bool: True if the solution is complete, False otherwise.
        """
        return self.not_colored == []

    def copy_solution(self) -> Self:
        """Creates a deep copy of the solution.

        Returns:
            Self: A deep copy of the solution.
        """
        return deepcopy(self)  # TODO more efficient copy

    @property
    def is_feasible(self) -> bool:
        """Checks if the solution is feasible (complete and no conflicts).

        Returns:
            bool: True if the solution is feasible, False otherwise.
        """
        return self.is_complete() and self.conflicts() == 0

    def colors_around(self, node: int) -> list[int]:
        """Returns a list of colors used by the neighbors of a given node.

        Args:
            node (int): The ID of the node.

        Returns:
            list[int]: A list of colors used by the neighbors of the node.
        """
        return [
            self.colors[neigh]
            for neigh in self.problem.g.neighbors(node)
            if self.colors[neigh] is not None
        ]


# ---------------------------------- Problem --------------------------------


@final
class Problem(
    SupportsConstructionNeighbourhood[AddNeighbourhood],
    SupportsEmptySolution[Solution],
    SupportsLocalNeighbourhood[OneRecolorNeighbourhood],
):
    def __init__(self, G: networkx.Graph, name: str, conflict_penalty=2):
        """Initializes the graph coloring problem instance.

        Args:
            G (networkx.Graph): The input graph.
            name (str): The name of the problem.
            conflict_penalty (int, optional): The penalty for each conflict in the objective function. Defaults to 2.
        """
        self.name = name
        self.c_nbhood: Optional[AddNeighbourhood] = None
        self.l_nbhood: Optional[OneRecolorNeighbourhood] = None
        mapping = {old: old - 1 for old in G.nodes()}
        G_relabelled = networkx.relabel_nodes(G, mapping)
        self.g = G_relabelled
        self.conflict_penalty = conflict_penalty

    def __str__(self) -> str:
        """Returns a string representation of the problem instance.

        Returns:
            str: A string describing the problem instance.
        """
        return f"Graph coloring problem {self.name} with {self.g.number_of_nodes()} nodes and {self.g.number_of_edges()} edges."

    def construction_neighbourhood(self) -> AddNeighbourhood:
        """Creates the construction neighbourhood for the problem.

        Returns:
            AddNeighbourhood: The construction neighbourhood.
        """
        if self.c_nbhood is None:
            self.c_nbhood = AddNeighbourhood(self)
        return self.c_nbhood

    def local_neighbourhood(self) -> OneRecolorNeighbourhood:
        """Creates the local neighbourhood for the problem.

        Returns:
            OneRecolorNeighbourhood: The local neighbourhood.
        """
        if self.l_nbhood is None:
            self.l_nbhood = OneRecolorNeighbourhood(self)
        return self.l_nbhood

    def empty_solution(self) -> Solution:
        """Creates an empty solution for the problem where all nodes are uncolored.

        Returns:
            Solution: An empty solution.
        """
        return Solution(self, [None] * len(self.g), 0)  # TODO better initial lb


def arg_parse():
    """Parse command line arguments"""
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "--inputFile",
        type=str,
        help="Input file path of graph problem",
        default="problems/graph-coloring/data/1-FullIns_3/1-FullIns_3.col",
    )
    group = argParser.add_mutually_exclusive_group()
    group.add_argument(
        "--sa", action="store_true", help="Run simulated annealing (local search)"
    )
    group.add_argument("--rls", action="store_true", help="Run random local search")
    argParser.add_argument(
        "--time", type=int, default=10, help="Time limit for local search"
    )
    argParser.add_argument(
        "--initial_temp",
        type=float,
        default=50,
        help="Initial temperature for simulated annealing",
    )
    argParser.add_argument(
        "--conflict_penalty",
        type=float,
        default=2.0,
        help="Penalty for each conflict in the objective function",
    )
    argParser.add_argument(
        "--outputFile",
        type=str,
        default=None,
        help="Output file path to save the solution",
    )

    return argParser.parse_args()


if __name__ == "__main__":
    # Parse the input file
    args = arg_parse()

    # Parsing the input file
    G = IOParser.parse2nx(args.inputFile)

    # Create the problem instance
    problem = Problem(G, "Graph coloring", conflict_penalty=args.conflict_penalty)

    # Run greedy construction to get an initial solution
    print("Starting greedy construction")
    gSolution = alg.greedy_construction(problem)
    print("Greedy construction finished, result:")
    Gresult = gSolution.to_textio()
    print(Gresult)

    if args.sa:
        # Run simulated annealing to improve the previous solution
        print("Starting simulated annealing")
        SAsolution = alg.sa(problem, gSolution, args.time, args.initial_temp)
        print("Simulated annealing finished, result:")
        SAresult = SAsolution.to_textio()
        print(SAresult)

    if args.rls:
        # Run random local search to improve the previous solution
        print("Starting random local search")
        RLSsolution = alg.rls(problem, gSolution, args.time)
        print("Random local search finished, result:")
        RLSresult = RLSsolution.to_textio()
        print(RLSresult)

    if args.outputFile:
        with open(args.outputFile, "w") as f:
            if args.sa:
                f.write(SAresult)
            elif args.rls:
                f.write(RLSresult)
            else:
                f.write(Gresult)
