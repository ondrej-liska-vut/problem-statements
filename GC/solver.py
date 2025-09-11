from __future__ import annotations
from copy import deepcopy
import networkx
from typing import Optional, Self, final
from collections import defaultdict
from constructive_search import AddNeighbourhood
from local_search import OneRecolorNeighbourhood
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


if __name__ == "__main__":
    import roar_net_api.algorithms as alg
    from parser import IOParser

    # name = "1-Insertions_4"
    # name = "0-SmallExample"
    name = "1-FullIns_4"
    G = IOParser.parse2nx(f"problems/graph-coloring/data/{name}/{name}.col")
    problem = Problem(G, name)

    # Run greedy construction to get an initial solution
    g1solution = alg.greedy_construction(problem)
    g2solution = alg.greedy_construction(problem)
    g3solution = alg.greedy_construction(problem)
    g4solution = alg.greedy_construction(problem)
    # solution = alg.beam_search(problem, bw=10)
    # solution = alg.grasp(problem, 30.0)

    # Run simulated annealing to improve the previous solution
    # SAsolution = alg.sa(problem, g1solution, 600.0, 1000.0)
    # print("Local search finished")
    # print(SAsolution.colors)
    # print(f"Local search finished with objective value {SAsolution.objective_value()}")

    # SA2solution = alg.sa(problem, g3solution, 600.0, 400.0)
    # print("Local search finished")
    # print(SA2solution.colors)
    # print(f"Local search finished with objective value {SA2solution.objective_value()}")

    # SA3solution = alg.sa(problem, g4solution, 600.0, 50.0)
    # print("Local search finished")
    # print(SA3solution.colors)
    # print(f"Local search finished with objective value {SA3solution.objective_value()}")

    # RLSsolution = alg.rls(problem, g2solution, 600)
    # print("Local search finished")
    # print(RLSsolution.colors)
    # print(f"Local search finished with objective value {RLSsolution.objective_value()}")

    greedy = alg.greedy_construction(problem)
    print("Greedy constructive  search finished")
    print(greedy.colors)
    print(
        f"Greedy constructive search finished with objective value {greedy.objective_value()}"
    )

    # g1solution = alg.greedy_construction(problem)
    # g2solution = alg.greedy_construction(problem)
    # g3solution = alg.greedy_construction(problem)
    # g4solution = alg.greedy_construction(problem)
    # solution = alg.beam_search(problem, bw=10)
    # solution = alg.grasp(problem, 30.0)

    # Run simulated annealing to improve the previous solution
    # SAsolution = alg.sa(problem, g1solution, 1200.0, 500.0)
    # print("Local search finished")
    # print(SAsolution.colors)
    # print(f"Local search finished with objective value {SAsolution.objective_value()}")

    # SA2solution = alg.sa(problem, g3solution, 1200.0, 300.0)
    # print("Local search finished")
    # print(SA2solution.colors)
    # print(f"Local search finished with objective value {SA2solution.objective_value()}")

    # SA3solution = alg.sa(problem, g4solution, 1200.0, 50.0)
    # print("Local search finished")
    # print(SA3solution.colors)
    # print(f"Local search finished with objective value {SA3solution.objective_value()}")

    # greedy = alg.greedy_construction(problem)
    # print("Greedy constructive  search finished")
    # print(greedy.colors)
    # print(
    #     f"Greedy constructive search finished with objective value {greedy.objective_value()}"
    # )
    # Print the final solution to stdout
    # solution.to_textio(sys.stdout)
