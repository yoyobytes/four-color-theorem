# DSATUR-UP-RC: Planar Graph 4-Coloring

**Best known backtracking-free polynomial-time algorithm for 4-coloring planar graphs.**

100% 4-coloring on tested planar graphs up to n≈300 (k=10 restarts).
Proven O(n²) per run; conjectured O(n) under the Bounded Propagation Depth Conjecture.

## Quick Start

```bash
pip install -r requirements.txt
python gui.py
# Open http://localhost:5050 in your browser
```

The GUI has three panels:
- **Visualize** — step-by-step animation of the algorithm on any graph (Delaunay, Errera, Kittell, Dodecahedron, Grid, Wheel, up to n=500)
- **How It Works** — explains DSATUR selection, unit propagation, 2-hop recolor, and the open conjecture
- **Benchmark** — live benchmark + comparison table against all known algorithms

## Algorithm

**DSATUR + Unit Propagation + 2-hop Recolor (DSATUR-UP-RC)**

1. **DSATUR selection** — pick the most constrained uncolored vertex (max forbidden colors)
2. **Unit propagation** — try each color on a shadow copy; cascade forced assignments; commit on success
3. **2-hop recolor** — if all colors fail, recolor a 1- or 2-hop neighbor to free a slot
4. **Restarts** — k independent passes, keep best result

## Key Results

| Algorithm | 4-colored | Type | Complexity |
|---|---|---|---|
| Greedy largest-first | 14.7% | greedy | O(n²) |
| Greedy smallest-last | 39.3% | greedy | O(n²) |
| RLF (Brown 1972) | 18.4% | greedy | O(n³) |
| DSATUR (Brélaz 1979) | 46.4% | greedy | O(n²) |
| TabuCol (H&deW 1987) | 93.6% | local search | O(iter·n) |
| **DSATUR-UP-RC k=10** | **100.0%** | **greedy+UP** | **O(n²) proven** |

Tested on 500 random planar Delaunay triangulations, n=15–80.

## Bounded Propagation Depth Conjecture

Across 11,000+ graphs (n up to 1,000), the depth of the unit propagation
tree at any contradiction is at most **14**, independent of n.
If proven: algorithm is O(n), faster than Robertson et al. O(n²),
without needing 633 reducible configurations.

## Paper

See `paper_new.pdf` for the full paper with proofs.

## Authors

- Yosef Ali Jiménez Muñoz (Universidad Tecnológica de Altamira, Programa Delfín 2026)
- Dr. José Raymundo Marcial-Romero (UAEM)
