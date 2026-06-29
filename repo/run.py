"""
run.py
======
Four Color Theorem Visualizer — main entry point.

Usage:
    python run.py                        # interactive menu
    python run.py --tabu errera          # tabu coloring on a named graph
    python run.py --pipeline errera      # full pipeline (greedy+Kempe+backtrack)
    python run.py --list                 # list all available graphs
    python run.py --help-format          # print the JSON format spec
"""

import argparse, sys
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Four Color Theorem Visualizer",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--tabu",        metavar="GRAPH", help="Run tabu coloring on a named graph or JSON path")
parser.add_argument("--pipeline",    metavar="GRAPH", help="Run full pipeline (greedy+Kempe+backtrack)")
parser.add_argument("--list",        action="store_true", help="List all available example graphs")
parser.add_argument("--help-format", action="store_true", help="Print the JSON input format")
parser.add_argument("--fps",         type=float, default=0.9, help="GIF frames per second (default 0.9)")
parser.add_argument("--output-dir",  default="output", help="Output directory (default: output/)")
args = parser.parse_args()

if args.help_format:
    print("""
JSON Graph Format
─────────────────
{
  "vertices": {"0": [x, y], "1": [x, y], ...},
  "aristas":  [[0, 1], [1, 2], ...]
}

  vertices  — node ID (string) → [x, y] coordinates (used for layout hints)
  aristas   — list of [u, v] integer pairs (undirected edges)

Coordinates are only used for the initial planarity check and maximalization.
The actual layout is recomputed by Tutte embedding.
Run with --list to see all built-in examples.
""")
    sys.exit(0)

from tabu_coloring import GRAPHS, render_gif, _print_list
import json, networkx as nx

def _load(name_or_path):
    """Load a graph by name from the registry or by path from a JSON file."""
    p = Path(name_or_path)
    if p.exists() and p.suffix == ".json":
        with open(p) as f: data = json.load(f)
        G = nx.Graph()
        pos = {}
        for vid, coords in data["vertices"].items():
            G.add_node(int(vid)); pos[int(vid)] = tuple(coords)
        for a, b in data["aristas"]: G.add_edge(int(a), int(b))
        return G, pos, p.stem
    # Check examples/ folder
    ep = Path("examples") / f"{name_or_path}.json"
    if ep.exists():
        with open(ep) as f: data = json.load(f)
        G = nx.Graph()
        pos = {}
        for vid, coords in data["vertices"].items():
            G.add_node(int(vid)); pos[int(vid)] = tuple(coords)
        for a, b in data["aristas"]: G.add_edge(int(a), int(b))
        return G, pos, name_or_path
    # Built-in registry
    if name_or_path in GRAPHS:
        fn, _ = GRAPHS[name_or_path]
        G, pos = fn()
        return G, pos, name_or_path
    raise KeyError(f"Unknown graph '{name_or_path}'. Use --list to see options.")

if args.list:
    _print_list()
    # Also show JSON examples
    extra = sorted(p.stem for p in Path("examples").glob("*.json")
                   if p.stem not in GRAPHS)
    if extra:
        print(f"JSON examples in examples/ ({len(extra)} graphs):")
        for name in extra: print(f"  {name}")
    sys.exit(0)

if args.tabu:
    G, pos, name = _load(args.tabu)
    out = str(Path(args.output_dir) / f"{name}_tabu.gif")
    render_gif(args.tabu if args.tabu in GRAPHS else None,
               output_path=out, fps=args.fps,
               G_override=G, pos_override=pos, name_override=name)
    sys.exit(0)

if args.pipeline:
    print("Pipeline mode: run onion_peeling.py directly.")
    print("  python onion_peeling.py --example errera")
    sys.exit(0)

# ── Interactive menu ──────────────────────────────────────────────────────────
def menu():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║     Four Color Theorem Visualizer                ║")
    print("║     github.com/yoyobytes/four-color-theorem      ║")
    print("╚══════════════════════════════════════════════════╝\n")

    while True:
        print("  1  Tabu coloring — built-in graph")
        print("  2  Tabu coloring — load JSON file")
        print("  3  Full pipeline (greedy + Kempe + backtrack)")
        print("  4  List all graphs")
        print("  5  Exit\n")

        choice = input("  Choose [1-5]: ").strip()

        if choice == "1":
            _print_list()
            key = input("  Graph name: ").strip()
            if not key: continue
            out = input(f"  Output [{key}_tabu.gif]: ").strip() or f"{key}_tabu.gif"
            Path(args.output_dir).mkdir(exist_ok=True)
            out = str(Path(args.output_dir) / out)
            try:
                G, pos, name = _load(key)
                render_gif(None, output_path=out, fps=args.fps,
                           G_override=G, pos_override=pos, name_override=name)
            except Exception as e:
                print(f"  Error: {e}")
            print()

        elif choice == "2":
            path = input("  JSON path: ").strip()
            if not path: continue
            try:
                G, pos, name = _load(path)
                out = str(Path(args.output_dir) / f"{name}_tabu.gif")
                Path(args.output_dir).mkdir(exist_ok=True)
                render_gif(None, output_path=out, fps=args.fps,
                           G_override=G, pos_override=pos, name_override=name)
            except Exception as e:
                print(f"  Error: {e}")
            print()

        elif choice == "3":
            print("  Run: python onion_peeling.py --example <name>")
            print("  or:  python onion_peeling.py --json <path>\n")

        elif choice == "4":
            _print_list()

        elif choice == "5":
            print("  Goodbye."); break

        else:
            print("  Invalid choice.\n")

if __name__ == "__main__":
    menu()
