from parser import IOParser
import networkx as nx
import json
import argparse

from networkx.algorithms.coloring import greedy_color


def solve(graph: nx.Graph, strategy: str = "saturation_largest_first") -> dict:
    """Solve the graph coloring problem using a greedy heuristic.

    Args:
        graph: A NetworkX graph object.
        strategy: A string indicating the coloring strategy.

    Returns:
        A dictionary mapping each node to its assigned color.
    """
    return greedy_color(graph, strategy=strategy)


def load_instance(path: str) -> nx.Graph:
    """Loads a DIMACS .col file into a NetworkX graph."""
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            if line.startswith("e"):
                _, u, v = line.strip().split()
                G.add_edge(int(u), int(v))
    return G


def write_solution(solution: dict, path: str):
    """Writes the solution to a file in ROAR-NET format (node color per line)."""
    with open(path, "w") as f:
        for node in sorted(solution):
            f.write(f"{node} {solution[node]}\n")


def main():


    G = IOParser.parse2nx("problems/graph_coloring/data/1-FullIns_3/1-FullIns_3.col")
    print(G)
    #coloring = solve(G, strategy=args.strategy)
    #write_solution(coloring, args.solution)


if __name__ == "__main__":
    main()
