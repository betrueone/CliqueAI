# max_clique_500.py
# Exact maximum clique solver for graphs up to ~500 vertices.
# Uses bitset blocks of 64-bit words per vertex and Tomita-style pruning with greedy coloring.

import json
from math import ceil
from typing import Any, List, Tuple
import random

from CliqueAI.protocol import MaximumCliqueOfLambdaGraph

WORD = (1 << 64) - 1

def blocks_for(n: int) -> int:
    return ceil(n / 64)

def bitmask(v: int) -> Tuple[int, int]:
    return v // 64, 1 << (v % 64)

def popcount_int(x: int) -> int:
    return x.bit_count()

def popcount_blocks(blocks: List[int]) -> int:
    return sum(b.bit_count() for b in blocks)

def copy_blocks(src: List[int]) -> List[int]:
    return src[:]  # shallow copy of ints

def blocks_and(a: List[int], b: List[int]) -> List[int]:
    return [x & y for x, y in zip(a, b)]

def blocks_or(a: List[int], b: List[int]) -> List[int]:
    return [x | y for x, y in zip(a, b)]

def blocks_minus(a: List[int], b: List[int]) -> List[int]:
    return [x & ~y for x, y in zip(a, b)]

def blocks_any_nonzero(a: List[int]) -> bool:
    for x in a:
        if x:
            return True
    return False

def blocks_iter_indices(a: List[int]) -> List[int]:
    res = []
    base = 0
    for blk in a:
        x = blk
        while x:
            l = x & -x
            idx = (l.bit_length() - 1)
            res.append(base + idx)
            x ^= l
        base += 64
    return res

class Graph:
    def __init__(self, n: int, edges: List[Tuple[int, int]] = None):
        assert n > 0 and n <= 500, "n must be 1..500 (adjust code if you need larger)"
        self.n = n
        self.b = blocks_for(n)
        self.adj = [ [0]*self.b for _ in range(n) ]
        if edges:
            for u, v in edges:
                self.add_edge(u, v)

    def add_edge(self, u: int, v: int):
        assert 0 <= u < self.n and 0 <= v < self.n and u != v
        bu, mu = bitmask(u)
        bv, mv = bitmask(v)
        self.adj[u][bv] |= mv
        self.adj[v][bu] |= mu

    def neighbors_blocks(self, v: int) -> List[int]:
        return self.adj[v]

# Greedy heuristic to get an initial large clique (lower bound)
def greedy_clique(graph: Graph) -> List[int]:
    n = graph.n
    order = list(range(n))
    random.shuffle(order)
    best = []
    for start in order:
        clique = [start]
        P = copy_blocks(graph.neighbors_blocks(start))
        for v in order:
            if v == start: continue
            bu, mv = bitmask(v)
            if P[bu] & mv:
                clique.append(v)
                P = blocks_and(P, graph.neighbors_blocks(v))
                if not blocks_any_nonzero(P):
                    break
        if len(clique) > len(best):
            best = clique[:]
    return best

# Greedy coloring on candidate set P_blocks, returns colors per vertex and max color.
# vertices: list of vertex indices (in some order) that are present in P_blocks.
def greedy_color(graph: Graph, P_blocks: List[int], vertices: List[int]) -> Tuple[List[int], int]:
    m = len(vertices)
    colors = [0]*m
    buckets = []  # list of bitmask of positions assigned to that color (as Python int positions)
    pos_bit = [1 << i for i in range(m)]

    # Build pos adjacency within P: for each position i, mask of other positions it conflicts with
    index_of = {v: i for i, v in enumerate(vertices)}
    pos_adj = [0]*m
    for i, v in enumerate(vertices):
        # neighbor set of v intersect P_blocks -> iterate indices and map to positions
        nb = graph.neighbors_blocks(v)
        mask = 0
        base = 0
        for blk in nb:
            x = blk & P_blocks[base//64] if False else 0  # placeholder, replaced below
            base += 64
        # Efficient walk: iterate over P's vertices and test adjacency
        for j, u in enumerate(vertices):
            bu, mv = bitmask(u)
            if graph.adj[v][bu] & mv:
                mask |= pos_bit[j]
        pos_adj[i] = mask

    # Assign colors greedily
    for i in range(m):
        assigned = False
        for c, bmask in enumerate(buckets):
            if (bmask & pos_adj[i]) == 0:
                buckets[c] |= pos_bit[i]
                colors[i] = c + 1
                assigned = True
                break
        if not assigned:
            buckets.append(pos_bit[i])
            colors[i] = len(buckets)
    return colors, len(buckets)

# Tomita-style recursive search using blocks
class MaxCliqueSolver:
    def __init__(self, graph: Graph):
        self.g = graph
        self.best_clique = greedy_clique(graph)
        # prepare full P
        self.fullP = [0]*self.g.b
        for v in range(self.g.n):
            blk, m = bitmask(v)
            self.fullP[blk] |= m

    # Helper: build vertex list from P_blocks ordered by descending degree-in-P
    def vertices_from_P(self, P_blocks: List[int]) -> List[int]:
        verts = blocks_iter_indices(P_blocks)
        # sort by degree in P descending (heuristic)
        degs = []
        for v in verts:
            # degree within P
            cnt = 0
            nb = self.g.adj[v]
            for a, b in zip(nb, P_blocks):
                cnt += (a & b).bit_count()
            degs.append(( -cnt, v))  # negative for descending
        degs.sort()
        return [v for _, v in degs]

    def expand(self, R_blocks: List[int], R_list: List[int], P_blocks: List[int]):
        if not blocks_any_nonzero(P_blocks):
            if len(R_list) > len(self.best_clique):
                self.best_clique = R_list[:]
            return

        vertices = self.vertices_from_P(P_blocks)
        if not vertices:
            if len(R_list) > len(self.best_clique):
                self.best_clique = R_list[:]
            return

        colors, maxc = greedy_color(self.g, P_blocks, vertices)

        # process vertices in reverse (Tomita)
        for idx in range(len(vertices)-1, -1, -1):
            v = vertices[idx]
            color = colors[idx]
            if len(R_list) + color <= len(self.best_clique):
                return
            # include v
            R_blocks_new = blocks_and(R_blocks, self.fullP)  # copy shape
            for i in range(self.g.b):
                R_blocks_new[i] = R_blocks[i]
            bv, mv = bitmask(v)
            R_blocks_new[bv] |= mv
            R_list.append(v)
            P_blocks_new = blocks_and(P_blocks, self.g.adj[v])
            self.expand(R_blocks_new, R_list, P_blocks_new)
            R_list.pop()
            # remove v from P_blocks
            P_blocks[bv] &= ~mv
            if not blocks_any_nonzero(P_blocks):
                break

    def max_clique(self) -> List[int]:
        R_blocks = [0]*self.g.b
        P_blocks = copy_blocks(self.fullP)
        self.expand(R_blocks, [], P_blocks)
        return sorted(self.best_clique)

# Example usage
if __name__ == "__main__":
    import os

    def read_all_data_files(folder: str = "data") -> list:
        files_data = []
        if not os.path.isdir(folder):
            print(f"Folder '{folder}' does not exist.")
            return files_data
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath) and fname.endswith(".json"):
                with open(fpath, "r") as f:
                    try:
                        files_data.append(json.load(f))
                    except Exception as e:
                        print(f"Failed to read {fname}: {e}")

        print(f"Loaded {len(files_data)} data files.")
        return files_data

    sample_data = read_all_data_files("data")
    for data in sample_data:
        adjacency_list = data["adjacency_list"]
        n = data["number_of_nodes"]
        edges = [(u, v) for u in range(n) for v in adjacency_list[u]]
        g = Graph(n, edges)
        solver = MaxCliqueSolver(g)
        clique = solver.max_clique()
        answer = data["maximum_clique"]
        print(f"Clique: {clique}, Answer: {answer}")
        if len(clique) > len(answer):
            print("✅ best performance: ", len(clique), ", answer: ", len(answer))
        else:
            print("❌ worst performance: ", len(clique), ", answer: ", len(answer))
