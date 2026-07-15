"""
run.py  —  Four Color Theorem Visualizer
==========================================
Single entry point. Runs any coloring method on any built-in or custom graph.

Usage
-----
    python run.py                          # interactive menu
    python run.py --method tabu --graph errera
    python run.py --method bilayer --graph kittell
    python run.py --method forest --graph errera
    python run.py --list                   # list all graphs
    python run.py --list-methods           # list all methods
"""

import argparse, sys, json
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Four Color Theorem Visualizer")
parser.add_argument("--method", choices=["tabu","bilayer","forest","pipeline"],
                    help="Coloring method")
parser.add_argument("--graph",  metavar="NAME_OR_PATH",
                    help="Built-in graph name or path to JSON file")
parser.add_argument("--output", metavar="FILE",
                    help="Output GIF path (default: output/<method>_<graph>.gif)")
parser.add_argument("--fps",    type=float, default=0.9)
parser.add_argument("--list",        action="store_true", help="List all graphs")
parser.add_argument("--list-methods",action="store_true", help="List all methods")
args = parser.parse_args()

# ── Method registry ───────────────────────────────────────────────────────────
METHODS = {
    "tabu": {
        "desc": "Tabu (node-by-node outer-face)  — fast, ~22% 4-colored",
        "module": "tabu_coloring",
        "render": "render_gif",
    },
    "propagation": {
        "desc": "DSATUR + Unit Propagation + Recolor  — 99.3% 4-colored, new algorithm",
        "module": "propagation_coloring",
        "render": "render_gif",
    },
    "dsatur": {
        "desc": "Complete 4-color (DSATUR+recolor+CSP)  — 100% 4-colored, best algorithm",
        "module": "dsatur_coloring",
        "render": "render_gif",
    },
    "schnyder": {
        "desc": "Schnyder order + recolor  — best result, ~62% 4-colored",
        "module": "schnyder_coloring",
        "render": "render_gif",
    },
    "bilayer": {
        "desc": "Bi-Layer (De Ita Luna & Marcial-Romero)  — closure+PFE detection",
        "module": "bilayer_coloring",
        "render": "render_gif",
    },
    "forest": {
        "desc": "Forest Decomposition analysis  — shows phase structure F0–F4",
        "module": "forest_decomposition",
        "render": "render_gif",
    },
    "pipeline": {
        "desc": "Full pipeline (greedy + Kempe + CSP backtracking)  — always 4-colors",
        "module": None,   # handled separately
        "render": None,
    },
}

if args.list_methods:
    print("\nAvailable methods:\n")
    for k, v in METHODS.items():
        print(f"  {k:<12} {v['desc']}")
    print()
    sys.exit(0)

# ── Graph loader ──────────────────────────────────────────────────────────────
def load_graph(name_or_path):
    """Load by built-in name, examples/ JSON, or direct JSON path."""
    import networkx as nx

    # Direct path
    p = Path(name_or_path)
    if p.exists() and p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        G = nx.Graph()
        pos = {}
        for vid, coords in data["vertices"].items():
            G.add_node(int(vid)); pos[int(vid)] = tuple(coords)
        for a, b in data["aristas"]:
            G.add_edge(int(a), int(b))
        return G, pos, p.stem

    # examples/ folder
    ep = Path("examples") / f"{name_or_path}.json"
    if ep.exists():
        with open(ep) as f:
            data = json.load(f)
        G = nx.Graph(); pos = {}
        for vid, coords in data["vertices"].items():
            G.add_node(int(vid)); pos[int(vid)] = tuple(coords)
        for a, b in data["aristas"]:
            G.add_edge(int(a), int(b))
        return G, pos, name_or_path

    # Built-in registries — try each module
    for mod_name in ["tabu_coloring", "bilayer_coloring",
                     "forest_decomposition", "layer_bipartite_coloring"]:
        try:
            mod = __import__(mod_name)
            if hasattr(mod, "GRAPHS") and name_or_path in mod.GRAPHS:
                fn, _ = mod.GRAPHS[name_or_path]
                G, pos = fn()
                return G, pos, name_or_path
        except Exception:
            pass

    raise KeyError(f"Graph '{name_or_path}' not found. Use --list to see options.")


def all_graph_names():
    """Return sorted union of all built-in graph names."""
    names = set()
    for mod_name in ["tabu_coloring", "bilayer_coloring",
                     "forest_decomposition", "layer_bipartite_coloring"]:
        try:
            mod = __import__(mod_name)
            if hasattr(mod, "GRAPHS"):
                names.update(mod.GRAPHS.keys())
        except Exception:
            pass
    # Also scan examples/
    for p in sorted(Path("examples").glob("*.json")):
        names.add(p.stem)
    return sorted(names)


if args.list:
    import networkx as nx
    print("\nBuilt-in graphs:\n")
    for mod_name in ["tabu_coloring"]:
        try:
            mod = __import__(mod_name)
            for k, (fn, desc) in mod.GRAPHS.items():
                try:
                    G, _ = fn()
                    planar = nx.check_planarity(G)[0]
                    tag = "" if planar else "  [not planar — skipped]"
                    print(f"  {k:<24} {desc}{tag}")
                except Exception:
                    print(f"  {k:<24} (error loading)")
        except Exception:
            pass
    extra = sorted(p.stem for p in Path("examples").glob("*.json"))
    if extra:
        print(f"\nJSON examples in examples/ ({len(extra)} files):")
        for name in extra[:20]:
            print(f"  {name}")
        if len(extra) > 20:
            print(f"  ... and {len(extra)-20} more")
    print()
    sys.exit(0)

# ── Direct CLI run ────────────────────────────────────────────────────────────
def run_method(method, graph_name, output_path, fps):
    import importlib, networkx as nx
    Path("output").mkdir(exist_ok=True)

    if output_path is None:
        output_path = f"output/{method}_{graph_name}.gif"

    if method == "pipeline":
        print("\nPipeline method: use onion_peeling.py directly.")
        print(f"  python onion_peeling.py --example {graph_name}")
        return

    cfg = METHODS[method]
    mod = importlib.import_module(cfg["module"])
    render = getattr(mod, cfg["render"])

    try:
        G, pos, name = load_graph(graph_name)
    except KeyError as e:
        print(f"\n  {e}"); sys.exit(1)

    # Planarity check — skip non-planar
    if not nx.check_planarity(G)[0]:
        print(f"\n  Graph '{graph_name}' is not planar. Skipping.")
        return

    print(f"\n  Method: {method}  |  Graph: {graph_name}")
    render(graph_name=None, output_path=output_path, fps=fps,
           G_override=G, pos_override=pos, name_override=name)


if args.method and args.graph:
    run_method(args.method, args.graph, args.output, args.fps)
    sys.exit(0)

# ── Interactive menu ───────────────────────────────────────────────────────────
def menu():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║     Four Color Theorem Visualizer                    ║")
    print("║     github.com/yoyobytes/four-color-theorem          ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    while True:
        print("  Methods:")
        for i, (k, v) in enumerate(METHODS.items(), 1):
            print(f"  {i}  {k:<12} — {v['desc']}")
        print("  5  Exit\n")

        choice = input("  Choose method [1-5]: ").strip()
        if choice == "5": print("  Goodbye."); break

        method_keys = list(METHODS.keys())
        try:
            method = method_keys[int(choice)-1]
        except (ValueError, IndexError):
            print("  Invalid.\n"); continue

        # Show graphs
        import tabu_coloring as T
        print(f"\n  Graphs for method '{method}':\n")
        graph_list = list(T.GRAPHS.keys())
        for i, name in enumerate(graph_list, 1):
            _, desc = T.GRAPHS[name]
            print(f"  {i:2d}  {name:<24} {desc}")
        extra = sorted(p.stem for p in Path("examples").glob("*.json")
                       if p.stem not in T.GRAPHS)
        if extra:
            print(f"\n  + {len(extra)} JSON examples in examples/")
        print()

        g = input("  Graph name (or number, or path to JSON): ").strip()
        if not g: continue
        try:
            idx = int(g) - 1
            if 0 <= idx < len(graph_list):
                g = graph_list[idx]
        except ValueError:
            pass

        out = input(f"  Output [{method}_{g}.gif]: ").strip() or f"{method}_{g}.gif"
        Path("output").mkdir(exist_ok=True)
        out = f"output/{out}" if "/" not in out else out

        fps_str = input("  FPS [0.9]: ").strip()
        fps = float(fps_str) if fps_str else 0.9

        try:
            run_method(method, g, out, fps)
        except Exception as e:
            print(f"  Error: {e}")
        print()

if __name__ == "__main__":
    menu()
