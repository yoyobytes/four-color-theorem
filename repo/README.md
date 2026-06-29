# Four Color Theorem Visualizer

Animated step-by-step visualization of two algorithms for 4-coloring planar graphs:

- **Tabu coloring** — node-by-node, outer-face peeling, no backtracking
- **Full pipeline** — greedy + Kempe chains + CSP backtracking with MRV

Includes **77+ example graphs** covering polyhedra, grids, lattices, random triangulations, Kempe counterexamples, wheels, cycles, antiprisms, and more.

---

## Installation

```bash
git clone https://github.com/yoyobytes/four-color-theorem
cd four-color-theorem
pip install -r requirements.txt
```

## Quick start

```bash
# Interactive menu
python run.py

# Run tabu coloring on a specific graph
python run.py --tabu errera
python run.py --tabu dodecahedron
python run.py --tabu delaunay_30

# List all available graphs
python run.py --list

# Run tabu directly
python tabu_coloring.py errera
python tabu_coloring.py --list

# Full pipeline (greedy + Kempe + backtracking)
python onion_peeling.py --example errera
python onion_peeling.py --example kittell
python onion_peeling.py --json examples/dodecahedron.json
```

Output GIFs are saved to `output/`.

---

## Tabu Coloring Algorithm

At each step:
1. Compute the **outer face** of the current surviving subgraph (face traversal on the planar embedding, maximizing eccentricity sum)
2. Pick the outer-face node with the **fewest forbidden colors** (ties broken by node ID)
3. Assign the **minimum color** not used by any already-colored direct neighbor
4. Remove that node and repeat

No backtracking. Runs in O(V²) due to repeated planarity checks.

The "tabu" constraint: `forbidden(v) = {color(u) : u ∈ neighbors(v), u already colored}` — the colors of previously colored neighbors, which propagates outward layer by layer as the outer face shrinks.

---

## Full Pipeline

`onion_peeling.py` runs:

| Step | What | Why |
|------|------|-----|
| Boyer-Myrvold | Planarity check | O(V) |
| Maximalization | Add edges → triangulation | Simplifies coloring |
| Tutte embedding | Solve Laplacian system | Crossing-free layout |
| Topological peeling | Outer face → remove min-degree node | Greedy order |
| Greedy + Kempe | Color in reverse peel order | ≤ 6 colors guaranteed |
| CSP backtracking | MRV + forward checking | 4-color guarantee |

---

## Example Graphs (77+)

| Category | Graphs |
|----------|--------|
| Kempe counterexamples | `errera`, `kittell` |
| Platonic polyhedra | `dodecahedron`, `icosahedron`, `octahedron` |
| Wheels | `wheel_5` through `wheel_12` |
| Grids | `grid_2x6`, `grid_3x4`, `grid_4x5`, `grid_5x5`, ... |
| Triangular lattice | `tri_lattice_2x4`, `tri_lattice_3x3`, ... |
| Hexagonal lattice | `hex_lattice_2x3`, `hex_lattice_3x3`, ... |
| Cycles | `cycle_5` through `cycle_12` |
| Ladders | `ladder_4`, `ladder_5`, ..., `ladder_10` |
| Circular ladders | `circular_ladder_4` through `circular_ladder_8` |
| Antiprisms | `antiprism_4` through `antiprism_8` |
| Stars | `star_5`, `star_7`, `star_10`, `star_15` |
| Bipartite | `bipartite_2_4`, `bipartite_2_6`, `bipartite_2_8` |
| Delaunay triangulations | `delaunay_10` through `delaunay_60` |
| Sunflower, book, stacked | `sunflower`, `book_5`, `stacked_tri` |

Run `python run.py --list` to see all with descriptions.

---

## JSON Format

```json
{
  "vertices": {"0": [0.0, 4.0], "1": [2.0, 5.0]},
  "aristas":  [[0, 1], [1, 2]]
}
```

- `vertices`: node ID (string) → `[x, y]` coordinates
- `aristas`: list of `[u, v]` integer pairs (undirected edges)

Coordinates are used only as hints for the maximalization step. The actual layout is recomputed by Tutte embedding.

Drop any `.json` file into `examples/` and it will appear in the menu automatically.

---

## References

- Errera, A. (1921). *Du coloriage des cartes*. ULB.
- Kempe, A. B. (1879). Four colours problem. *American Journal of Mathematics*.
- Tutte, W. T. (1963). How to draw a graph. *Proc. London Mathematical Society*.
- Appel, K. & Haken, W. (1977). Every planar map is four colorable. *Illinois J. Mathematics*.
- Robertson, N. et al. (1997). The four-colour theorem. *J. Combinatorial Theory B*.
- Gonthier, G. (2008). Formal proof — the four-color theorem. *Notices of the AMS*.

---

## Bi-Layer Coloring (De Ita Luna & Marcial-Romero)

`bilayer_coloring.py` implements the Bi-Layer algorithm from the paper
*"A combinatorial algorithmic proof for the 4-coloring on planar graphs"*
(De Ita Luna & Marcial-Romero, BUAP, 2024).

Key features over plain greedy:
- **Failure detection**: failed vertices, edges, faces, wheels
- **Potential failure detection**: pfe (potential failed edges), pff (potential failed faces)
- **Closure T((vi,a))**: propagates forced colorings transitively to evaluate color choices
- **Stack**: vertices with deg + |tabu| ≤ 3 are deferred and colored last
- **Hierarchical vertex selection**: pfe nodes → complementary tabus → max degree
- **Closure-based color scoring**: minimize future failure configurations

```bash
python bilayer_coloring.py errera
python bilayer_coloring.py kittell
python bilayer_coloring.py --list
```

Note: current implementation is a partial implementation — the full global-pass
transitive closure is future work (see paper Section 10).

---

## Layer-Bipartite Coloring (Jiménez Muñoz, 2026)

`layer_bipartite_coloring.py` — a new experimental algorithm exploiting the
bipartite structure of inner peeling layers.

**Key observation:** when a planar graph is peeled layer by layer, inner layers
are often bipartite (2-colorable internally). This structure is exploited to
assign colors efficiently:

1. For each layer, check if it is bipartite.
2. If bipartite: find one color for all of part P (minimum color not forbidden
   by any P node's cross-layer neighbors). Then color Q greedily.
3. If not bipartite: fall back to greedy with tabu.

**Results vs tabu (node-by-node):**
- Kittell: Layer-Bipartite gets **4 colors** vs tabu's 5 — a win.
- Errera: Layer-Bipartite gets 5 colors (L1→L2 interaction creates a forced
  conflict). Combining with the Bi-Layer closure is the next research step.

**Open question:** does Layer-Bipartite + Bi-Layer closure always produce valid
4-colorings for all planar graphs?

```bash
python layer_bipartite_coloring.py errera
python layer_bipartite_coloring.py kittell
python layer_bipartite_coloring.py --compare errera
python layer_bipartite_coloring.py --list
```
