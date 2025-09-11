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
    def __init__(self, problem, colors: list[Optional[int]], lb: float, nodes_available_colors: list[list[int]] = None):
        self.problem = problem
        self.colors = colors
        self.not_colored = [i for i, c in enumerate(colors) if c is None]
        self.lb = lb
        self.used_colors = len({c for c in self.colors if c is not None})
        self.nodes_available_colors = nodes_available_colors if nodes_available_colors is not None else self.nodes_available_colors_method()
        self.color_map = defaultdict(list)
        self._objective_value = None
        for n, c in enumerate(colors):
            if c is not None:
                self.color_map[c].append(n)

    def nodes_max_available_colors(self) -> list[int]:
        max_available_colors = [[] for _ in range(len(self.problem.g.nodes))]
        for node in range(len(self.problem.g.nodes)):
            count = len(list(self.problem.g.neighbors(node))) + 1
            max_available_colors[node] = count
        return max_available_colors
    
    def nodes_available_colors_method(self) -> list[list[int]]:
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

    def to_textio(self) -> None:
        print("Solution:")
        print(" Colors of nodes:",self.colors)
        print(" Count of used colors:",self.used_colors)
        pass

    def conflicts(self) -> int:
        cols = self.colors
        cnt = 0
        for u, v in self.problem.g.edges:
            cu, cv = cols[u], cols[v]
            if cu is not None and cv is not None and cu == cv:
                cnt += 1
        return cnt

    def update_objective_value(self, value: Optional[float]) -> Optional[float]:
        self._objective_value = value
        return self._objective_value

    def objective_value(self) -> float:
        if self._objective_value is not None:
            return self._objective_value
        else:
            self._objective_value = (
                self.conflicts() * self.problem.conflict_penalty + self.used_colors
            )
            return self._objective_value

    def is_complete(self) -> bool:
        return self.not_colored == []

    def copy_solution(self) -> Self:
        return deepcopy(self)  # TODO more efficient copy

    @property
    def is_feasible(self) -> bool:
        return self.is_complete() and self.conflicts() == 0

    def colors_around(self, node: int) -> list[int]:
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
        self.name = name
        self.c_nbhood: Optional[AddNeighbourhood] = None
        self.l_nbhood: Optional[OneRecolorNeighbourhood] = None
        mapping = {old: old - 1 for old in G.nodes()}
        G_relabelled = networkx.relabel_nodes(G, mapping)
        self.g = G_relabelled
        self.conflict_penalty = conflict_penalty

    # def __str__(self) -> str:
    #     out: list[str] = []
    #     for row in self.dist:
    #         out.append(" ".join(map(str, row)))
    #     return "\n".join(out)

    def construction_neighbourhood(self) -> AddNeighbourhood:
        if self.c_nbhood is None:
            self.c_nbhood = AddNeighbourhood(self)
        return self.c_nbhood

    def local_neighbourhood(self) -> OneRecolorNeighbourhood:
        if self.l_nbhood is None:
            self.l_nbhood = OneRecolorNeighbourhood(self)
        return self.l_nbhood

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

def arg_parse():
    ''' Parse command line arguments'''
    argParser = argparse.ArgumentParser()
    argParser.add_argument("inputFile", type=str, help="Input file path of graph problem")
    argParser.add_argument("--sa", action="store_true", help="Run simulated annealing (local search)")
    argParser.add_argument("--rls", action="store_true", help="Run random local search")
    argParser.add_argument("--time", type=int, default=10, help="Time limit for local search")
    argParser.add_argument("--initial_temp", type=float, default=50, help="Initial temperature for simulated annealing")
    argParser.add_argument("--conflict_penalty", type=float, default=2.0, help="Penalty for each conflict in the objective function")
    argParser.add_argument("--outputFile", type=str, default=None, help="Output file path to save the solution")

    return argParser.parse_args() 

if __name__ == "__main__":
     # Parse the input file
    args = arg_parse()

    #Parsing the input file
    G = IOParser.parse2nx(args.inputFile)

    # Create the problem instance
    problem = Problem(G, "Graph coloring", conflict_penalty=args.conflict_penalty)

    # Run greedy construction to get an initial solution
    print("Starting greedy construction")
    gSolution = alg.greedy_construction(problem)
    print("Greedy construction finished")
    gSolution.to_textio()

    if(args.sa):
        # Run simulated annealing to improve the previous solution
        print("Starting simulated annealing")
        SAsolution = alg.sa(problem, gSolution, args.time, args.initial_temp)
        print("Simulated annealing finished")
        SAsolution.to_textio()

    if(args.rls):
        # Run random local search to improve the previous solution
        print("Starting random local search")
        RLSsolution = alg.rls(problem, gSolution, args.time)
        print("Random local search finished")
        RLSsolution.to_textio()