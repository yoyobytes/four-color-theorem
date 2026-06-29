"""
bilayer_coloring.py
===================
Implementation of the Bi-Layer coloring algorithm by De Ita Luna & Marcial-Romero.

Algorithm overview
------------------
At each step, one vertex from the outer face is selected and colored.
The key contributions over plain greedy are:

1. FAILURE DETECTION — recognize configurations that would force a 5th color:
   (a) Failed vertex:  |Tabu(v)| = 4
   (b) Failed edge:    Tabu(v1)=Tabu(v2), |Tabu|=3, {v1,v2} in E
   (c) Failed face:    Tabu(v1)=Tabu(v2)=Tabu(v3), |Tabu|=2, triangle

2. POTENTIAL FAILURE DETECTION — pre-configurations that lead to failure:
   - Potential failed edge (pfe): Tabu(v1)=Tabu(v2), |Tabu|=2,
     exists w adjacent to both
   - Potential failed face (pff): intersection of tabus non-empty,
     |Tabu(v1)|=2, exists w adj to v2,v3 but not v1

3. CLOSURE T((vi,a)) — propagating forced colorings:
   Assigning color a to vi may force some neighbors to take specific colors
   (if |Tabu(u)|=3, only one color remains → forced assignment).
   These propagate transitively. If the closure creates a failure config
   → INCONSISTENT (do not use this color).

4. STACK — safe vertex removal:
   If deg(v) + |Tabu(v)| <= 3, v can always be colored safely at the end.
   Push it to the stack and remove from the active graph.

5. VERTEX SELECTION (hierarchical):
   1st: vertices forming potential failed edges
   2nd: vertices with complementary tabus
   3rd: max degree in outer face

6. COLOR SELECTION:
   For each candidate color: run closure, evaluate virtual graph,
   pick color that minimizes failure pre-configurations.

Reference
---------
De Ita Luna, G. & Marcial-Romero, R. (2024).
"A combinatorial algorithmic proof for the 4-coloring on planar graphs."
Facultad de Ciencias de la Computación, BUAP.
"""

import math, sys
from itertools import combinations
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
import numpy as np

FOUR = {0, 1, 2, 3}
PAL  = {0:"#2196F3", 1:"#4CAF50", 2:"#FF9800", 3:"#E91E63"}
CNAMES = {0:"Blue", 1:"Green", 2:"Orange", 3:"Pink"}

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH LIBRARY (same as tabu_coloring.py)
# ══════════════════════════════════════════════════════════════════════════════

def _circle(nodes, r, a0=0.0):
    nodes = list(nodes); n = len(nodes)
    return {v: (round(r*math.cos(2*math.pi*i/n+a0), 4),
                round(r*math.sin(2*math.pi*i/n+a0), 4))
            for i, v in enumerate(nodes)}

def _spring(G, scale=7.0, seed=42):
    raw = nx.spring_layout(G, seed=seed, scale=scale)
    return {n: (round(float(x),4), round(float(y),4)) for n,(x,y) in raw.items()}

def graph_errera():
    "Errera (1921) — 17 nodes. 1st Kempe counterexample. χ=4"
    e=[]
    for i in range(1,6): e.append((0,i))
    for i in range(1,5): e.append((i,i+1))
    e.append((5,1))
    for i in range(5):
        b=i+1; e.append((b,6+i)); e.append((b,6+(i+1)%5))
    for i in range(6,10): e.append((i,i+1))
    e.append((10,6))
    for i in range(5):
        b=6+i; e.append((b,11+i)); e.append((b,11+(i+1)%5))
    for i in range(11,15): e.append((i,i+1))
    e.append((15,11))
    for i in range(11,16): e.append((16,i))
    G=nx.Graph(); G.add_edges_from(e)
    pos={0:(0,0),16:(0.3,0.3)}
    pos.update(_circle([1,2,3,4,5],2.5,math.pi/2))
    pos.update(_circle([6,7,8,9,10],5.0,math.pi/2+math.pi/5))
    pos.update(_circle([11,12,13,14,15],7.5,math.pi/2))
    return G, pos

def graph_kittell():
    "Kittell (1935) — 23 nodes. 2nd Kempe counterexample. χ=4"
    e=[]
    for i in range(10): e.append((i,(i+1)%10))
    for i in range(5):
        b=10+i; e.extend([(b,2*i),(b,(2*i+1)%10),(b,(2*i+2)%10)])
    for i in range(5): e.append((10+i,10+(i+1)%5))
    for i in range(5):
        inn=15+i; e.extend([(inn,10+i),(inn,10+(i+1)%5)])
    for i in range(5): e.append((15+i,15+(i+1)%5))
    e.extend([(20,21),(21,22),(22,20)])
    for i,c in enumerate([20,20,21,21,22]): e.append((15+i,c))
    e.extend([(22,19),(20,19)])
    G=nx.Graph(); G.add_edges_from(e)
    pos={}
    pos.update(_circle([20,21,22],1.5,math.pi/2))
    pos.update(_circle(range(15,20),3.5,math.pi/2))
    pos.update(_circle(range(10,15),5.5,math.pi/2))
    pos.update(_circle(range(10),8.5,math.pi/2))
    return G, pos

def graph_layered_42():
    "Layered 42 — 42 nodes, 5 concentric rings, planar"
    rings=[15,11,8,5,3]; e=[]; offset=0; offsets=[0]
    for r in rings:
        for i in range(r): e.append((offset+i, offset+(i+1)%r))
        offset+=r; offsets.append(offset)
    for k in range(len(rings)-1):
        r_out=rings[k]; r_in=rings[k+1]
        off_out=offsets[k]; off_in=offsets[k+1]
        for i in range(r_in):
            j=int(i*r_out/r_in)
            e.append((off_in+i, off_out+j))
            e.append((off_in+i, off_out+(j+1)%r_out))
    G=nx.Graph(); G.add_edges_from(e)
    pos={}
    radii=[9.5,7.0,5.0,3.0,1.2]
    for k,r in enumerate(rings):
        pos.update(_circle(range(offsets[k],offsets[k+1]),radii[k],math.pi/2))
    return G, pos

def graph_dodecahedron():
    "Dodecahedron — 20 nodes. 3-regular, χ=3"
    G=nx.convert_node_labels_to_integers(nx.dodecahedral_graph())
    return G, _spring(G)

def graph_icosahedron():
    "Icosahedron — 12 nodes. 5-regular, χ=4"
    G=nx.convert_node_labels_to_integers(nx.icosahedral_graph())
    return G, _spring(G)

def graph_octahedron():
    "Octahedron — 6 nodes. 4-regular, χ=3"
    G=nx.convert_node_labels_to_integers(nx.octahedral_graph())
    return G, _spring(G, scale=5)

def graph_wheel(k=8):
    G=nx.wheel_graph(k); G=nx.convert_node_labels_to_integers(G)
    pos={0:(0.0,0.0)}; pos.update(_circle(range(1,k),4.0,math.pi/2))
    return G, pos

def graph_grid_4x4():
    G=nx.convert_node_labels_to_integers(nx.grid_2d_graph(4,4))
    return G, _spring(G, scale=6)

def graph_sunflower():
    "Sunflower — 13 nodes. 6 petals. χ=3"
    k=6; e=[]
    for i in range(k):
        a=1+2*i; b=2+2*i; e.extend([(0,a),(0,b),(a,b)])
        e.append((b,1+2*((i+1)%k)))
    G=nx.Graph(); G.add_edges_from(e)
    pos={0:(0.0,0.0)}; pos.update(_circle(range(1,2*k+1),4.0,math.pi/2))
    return G, pos

GRAPHS = {
    "errera":       (graph_errera,     "Errera (17)      — 1st Kempe counterexample, χ=4"),
    "kittell":      (graph_kittell,    "Kittell (23)     — 2nd Kempe counterexample, χ=4"),
    "layered_42":   (graph_layered_42, "Layered 42 (42)  — 5 concentric rings, planar"),
    "dodecahedron": (graph_dodecahedron,"Dodecahedron (20)— 3-regular, χ=3"),
    "icosahedron":  (graph_icosahedron,"Icosahedron (12) — 5-regular, χ=4"),
    "octahedron":   (graph_octahedron, "Octahedron (6)   — 4-regular, χ=3"),
    "wheel_8":      (lambda: graph_wheel(8), "Wheel W₈ (8) — odd rim, χ=4"),
    "grid_4x4":     (graph_grid_4x4,  "Grid 4×4 (16)   — χ=2"),
    "sunflower":    (graph_sunflower,  "Sunflower (13)   — χ=3"),
}

# ══════════════════════════════════════════════════════════════════════════════
# PLANAR UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _outer_face(G):
    if G.number_of_nodes() <= 2: return set(G.nodes())
    ok, emb = nx.check_planarity(G)
    if not ok: return set(G.nodes())
    seen = set(); faces = []
    for v in emb:
        for w in emb.neighbors(v):
            f = tuple(emb.traverse_face(v, w)); k = frozenset(f)
            if k not in seen: seen.add(k); faces.append(list(f))
    try:
        ecc = nx.eccentricity(G)
        return set(max(faces, key=lambda f: sum(ecc[v] for v in set(f))))
    except:
        return set(G.nodes())

def _maximize(G):
    G2 = G.copy()
    cands = sorted([(u,v) for u,v in combinations(G2.nodes(),2) if not G2.has_edge(u,v)],
                   key=lambda e: G2.degree(e[0])+G2.degree(e[1]))
    for u,v in cands:
        G2.add_edge(u,v)
        if not nx.check_planarity(G2)[0]: G2.remove_edge(u,v)
    return G2

def _tutte(G, scale=8.0):
    ok, emb = nx.check_planarity(G)
    seen = set(); faces = []
    for v in emb:
        for w in emb.neighbors(v):
            f = tuple(emb.traverse_face(v,w)); k = frozenset(f)
            if k not in seen: seen.add(k); faces.append(list(set(f)))
    ecc = nx.eccentricity(G)
    outer = list(set(max(faces, key=lambda f: sum(ecc[v] for v in set(f)))))
    ko = len(outer)
    pf = {v: (scale*math.cos(2*math.pi*i/ko+math.pi/2),
               scale*math.sin(2*math.pi*i/ko+math.pi/2))
          for i,v in enumerate(outer)}
    inn = [v for v in sorted(G.nodes()) if v not in pf]
    if not inn: return pf
    idx = {v:i for i,v in enumerate(inn)}; m = len(inn)
    A = np.zeros((m,m)); bx = np.zeros(m); by = np.zeros(m)
    for i,v in enumerate(inn):
        nb = list(G.neighbors(v)); A[i,i] = len(nb)
        for u in nb:
            if u in idx: A[i,idx[u]] = -1
            else: bx[i]+=pf[u][0]; by[i]+=pf[u][1]
    try: xc=np.linalg.solve(A,bx); yc=np.linalg.solve(A,by)
    except: return {**pf,**{v:(0.0,0.0) for v in inn}}
    pos=dict(pf)
    for i,v in enumerate(inn): pos[v]=(float(xc[i]),float(yc[i]))
    return pos

def _tri_faces(G):
    ok,emb=nx.check_planarity(G)
    if not ok: return []
    seen=set(); faces=[]
    for v in emb:
        for w in emb.neighbors(v):
            f=tuple(emb.traverse_face(v,w)); k=frozenset(f)
            if k not in seen:
                seen.add(k); face=list(set(f))
                if len(face)==3: faces.append(face)
    return faces

# ══════════════════════════════════════════════════════════════════════════════
# BI-LAYER ALGORITHM CORE
# ══════════════════════════════════════════════════════════════════════════════

def tabu(v, color, G):
    """Colors of already-colored neighbors of v."""
    return {color[u] for u in G.neighbors(v) if u in color}

def closure(vi, a, color, G):
    """
    T((vi, a)): reflexive-transitive closure of forced colorings.

    Step 0 (global pass): find ALL uncolored nodes already forced
    (only one color remaining) given current color + {vi:a}.
    Repeat until fixpoint — catches non-adjacent forced nodes.

    Returns (forced_dict, consistent).
    """
    forced = {vi: a}

    def collect_tabu(u):
        t = {color[nb] for nb in G.neighbors(u) if nb in color}
        t |= {forced[nb] for nb in G.neighbors(u) if nb in forced}
        return t

    # Reflexive-transitive global pass until fixpoint
    changed = True
    while changed:
        changed = False
        for u in G.nodes():
            if u in color or u in forced:
                continue
            t_u = collect_tabu(u)
            remaining = FOUR - t_u
            if len(remaining) == 0:
                return forced, False          # failed vertex
            if len(remaining) == 1:
                fc = next(iter(remaining))
                forced[u] = fc
                changed = True
                # Check immediately for adjacent forced conflict
                for nb in G.neighbors(u):
                    if nb in forced and forced[nb] == fc and nb != u:
                        return forced, False

    # Validate: no two adjacent forced nodes share a color
    for u, w in G.edges():
        if u in forced and w in forced and forced[u] == forced[w]:
            return forced, False
    # Validate: no forced node conflicts with existing coloring
    all_col = {**color, **forced}
    for u, w in G.edges():
        if u in all_col and w in all_col and all_col[u] == all_col[w]:
            return forced, False

    return forced, True

def detect_failures(G, color):
    """Detect failure configurations in current graph state."""
    failures = {"vertices": [], "edges": [], "faces": []}
    for v in G.nodes():
        if v in color: continue
        t = tabu(v, color, G)
        if len(t) == 4:
            failures["vertices"].append(v)
    for u,v in G.edges():
        if u in color or v in color: continue
        tu = tabu(u, color, G); tv = tabu(v, color, G)
        if tu == tv and len(tu) == 3:
            failures["edges"].append((u,v))
    for face in _tri_faces(G.subgraph([n for n in G.nodes() if n not in color])):
        tabs = [tabu(v, color, G) for v in face]
        if tabs[0]==tabs[1]==tabs[2] and len(tabs[0])==2:
            failures["faces"].append(tuple(face))
    return failures

def detect_pfe(G, color):
    """
    Potential Failed Edges: {v1,v2} where
    Tabu(v1)=Tabu(v2), |Tabu|=2, exists w adjacent to both.
    """
    pfes = []
    uncolored = [v for v in G.nodes() if v not in color]
    for u, v in combinations(uncolored, 2):
        tu = tabu(u, color, G); tv = tabu(v, color, G)
        if tu == tv and len(tu) == 2:
            # Check if common neighbor exists
            nu = set(G.neighbors(u)) - {v}
            nv = set(G.neighbors(v)) - {u}
            if nu & nv:
                pfes.append((u, v, sorted(tu), sorted(nu & nv)))
    return pfes

def detect_complementary_tabus(G, color):
    """
    Vertices with complementary tabus at distance 2:
    |Tabu(v1)|=2, |Tabu(v2)|=2, Tabu(v1)∪Tabu(v2)=Four,
    dist(v1,v2)=2.
    """
    uncolored = [v for v in G.nodes() if v not in color]
    comps = []
    for u, v in combinations(uncolored, 2):
        tu = tabu(u, color, G); tv = tabu(v, color, G)
        if len(tu)==2 and len(tv)==2 and tu|tv==FOUR:
            # Check distance 2
            try:
                d = nx.shortest_path_length(G.subgraph(uncolored), u, v)
                if d == 2:
                    comps.append((u, v))
            except: pass
    return comps

def count_pre_failures(G, color):
    """Count all pre-failure indicators (for color selection scoring)."""
    pfes  = detect_pfe(G, color)
    comps = detect_complementary_tabus(G, color)
    fails = detect_failures(G, color)
    # Count nodes that would be failed or in danger given current coloring
    future_failed  = [v for v in G.nodes() if v not in color
                      and len(tabu(v, color, G)) == 4]
    future_danger  = [v for v in G.nodes() if v not in color
                      and len(tabu(v, color, G)) == 3]
    return {
        "future_failed": len(future_failed),
        "fail_v":        len(fails["vertices"]) + len(future_failed),
        "fail_e":        len(fails["edges"]),
        "fail_f":        len(fails["faces"]),
        "pfe":           len(pfes),
        "danger":        len(future_danger),
        "comps":         len(comps),
    }

def _score_tuple(score):
    """Primary sort key for color selection — lower is better."""
    return (score["future_failed"], score["fail_v"], score["fail_e"],
            score["fail_f"], score["pfe"], score["danger"], score["comps"])

def select_vertex(outer_face, G, color):
    """
    Hierarchical vertex selection from outer face:
    1. Vertices in potential failed edges
    2. Vertices with complementary tabus (dist 2)
    3. Max degree vertex
    """
    uncolored_outer = [v for v in outer_face if v not in color]
    if not uncolored_outer:
        return None

    # 1. Check for pfe involvement
    pfes = detect_pfe(G, color)
    pfe_nodes = {v for pfe in pfes for v in pfe[:2]}
    pfe_outer = [v for v in uncolored_outer if v in pfe_nodes]
    if pfe_outer:
        return min(pfe_outer, key=lambda v: (len(FOUR - tabu(v, color, G)), -G.degree(v)))

    # 2. Complementary tabus
    comps = detect_complementary_tabus(G, color)
    comp_nodes = {v for pair in comps for v in pair}
    comp_outer = [v for v in uncolored_outer if v in comp_nodes]
    if comp_outer:
        return min(comp_outer, key=lambda v: (len(FOUR - tabu(v, color, G)), -G.degree(v)))

    # 3. Max degree
    return max(uncolored_outer, key=lambda v: G.degree(v))

def select_color(vi, G, color):
    """
    Color selection via closure:
    For each candidate color, run closure and score the virtual graph.
    Pick the color with minimum pre-failure configurations.
    Return (chosen_color, closure_forced, is_consistent, scores).
    """
    t = tabu(vi, color, G)
    candidates = sorted(FOUR - t)

    if not candidates:
        return None, {}, False, {}

    best_color = None
    best_forced = {}
    best_score = None
    results = []

    for a in candidates:
        forced, consistent = closure(vi, a, color, G)
        if not consistent:
            results.append((a, forced, False, None))
            continue
        # Virtual graph: apply forced colorings and score
        virtual_color = {**color, **forced}
        score = count_pre_failures(G, virtual_color)
        score_tuple = _score_tuple(score)
        results.append((a, forced, True, score_tuple))

        if best_score is None or score_tuple < best_score:
            best_score = score_tuple
            best_color = a
            best_forced = forced

    if best_color is None:
        # All closures inconsistent — fall back to min tabu
        best_color = candidates[0]
        best_forced = {vi: best_color}

    return best_color, best_forced, True, results

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

def bilayer_coloring(G_orig):
    """
    Run the Bi-Layer coloring algorithm.
    Returns (color, steps) where steps is a list of frame dicts.
    """
    G = G_orig.copy()
    color = {}
    stack = []
    steps = []

    def push_safe():
        """Push vertices with deg(G)+|tabu|<=3 to stack iteratively."""
        changed = True
        while changed:
            changed = False
            for v in list(G.nodes()):
                if v in color: continue
                t = tabu(v, color, G_orig)
                if G.degree(v) + len(t) <= 3:
                    stack.append(v)
                    G.remove_node(v)
                    changed = True
                    break

    while G.number_of_nodes() > 0:
        push_safe()
        if G.number_of_nodes() == 0:
            break

        of = _outer_face(G)
        vi = select_vertex(of, G_orig, color)
        if vi is None:
            vi = next((v for v in G.nodes() if v not in color), None)
        if vi is None:
            break

        a, forced, consistent, results = select_color(vi, G_orig, color)
        if a is None:
            t = tabu(vi, color, G_orig)
            avail = FOUR - t
            a = min(avail) if avail else 0
            forced = {vi: a}

        # Validate forced assignments — skip any that conflict
        clean_forced = {}
        for fv, fc in forced.items():
            conflict = any(color.get(nb) == fc
                           for nb in G_orig.neighbors(fv) if nb != fv)
            if not conflict:
                clean_forced[fv] = fc
            elif fv == vi:
                # Must color vi — find non-conflicting color
                t2 = tabu(fv, color, G_orig)
                avail = FOUR - t2
                clean_forced[fv] = min(avail) if avail else fc
        forced = clean_forced
        if vi not in forced:
            t = tabu(vi, color, G_orig)
            avail = FOUR - t
            forced[vi] = min(avail) if avail else 0
        a = forced[vi]

        pfes_before  = detect_pfe(G_orig, color)
        comps_before = detect_complementary_tabus(G_orig, color)
        fails_before = detect_failures(G_orig, color)

        color.update(forced)

        pfes_after  = detect_pfe(G_orig, color)
        fails_after = detect_failures(G_orig, color)

        pfe_nodes  = {v for pfe in pfes_before for v in pfe[:2]}
        comp_nodes = {v for pair in comps_before for v in pair}

        steps.append({
            "victim":          vi,
            "color":           a,
            "forced":          dict(forced),
            "outer":           set(of),
            "color_snap":      dict(color),
            "alive":           list(G.nodes()),
            "stack":           list(stack),
            "pfe_nodes":       pfe_nodes,
            "comp_nodes":      comp_nodes,
            "pfes_before":     pfes_before,
            "fails_before":    fails_before,
            "fails_after":     fails_after,
            "pfes_after":      pfes_after,
            "closure_results": results,
        })
        G.remove_node(vi)

    # Color stack vertices safely
    for v in reversed(stack):
        t = tabu(v, color, G_orig)
        c = 0
        while c in t: c += 1
        color[v] = c
        steps.append({
            "victim": v, "color": c, "forced": {v: c},
            "outer": set(), "color_snap": dict(color),
            "alive": [], "stack": list(stack),
            "pfe_nodes": set(), "comp_nodes": set(),
            "pfes_before": [], "fails_before": {},
            "fails_after": {}, "pfes_after": [],
            "closure_results": [], "from_stack": True,
        })

    return color, steps

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

C_BG="#FAFAFA"; C_PANEL="#ECEFF1"; C_EDGE="#607D8B"
C_GHOST="#E0E0E0"; C_OUT="#FB8C00"; C_PFE="#FF1744"
C_COMP="#7B1FA2"; C_FORCED="#FFA000"; C_STACK="#26C6DA"
C_GRAY="#90A4AE"

def render_gif(graph_name=None, output_path=None, fps=0.7,
               G_override=None, pos_override=None, name_override=None):
    if G_override is not None:
        G_orig, pos_orig = G_override, pos_override
        graph_name = name_override or "custom"
    elif graph_name and graph_name in GRAPHS:
        fn, _ = GRAPHS[graph_name]
        G_orig, pos_orig = fn()
    else:
        raise KeyError(f"Unknown graph '{graph_name}'.")

    print(f"\n  {graph_name}: {G_orig.number_of_nodes()}V {G_orig.number_of_edges()}E")

    if not nx.check_planarity(G_orig)[0]:
        print("  ✗ Not planar."); return

    print("  Maximalizing...")
    G_max = _maximize(G_orig)
    print(f"  Computing layout...")
    pos   = _tutte(G_max)
    tris  = _tri_faces(G_max)

    print("  Running Bi-Layer algorithm...")
    color, steps = bilayer_coloring(G_max)

    nc    = len(set(color.values()))
    valid = all(color.get(u,-1)!=color.get(v,-1) for u,v in G_max.edges()
                if u in color and v in color)
    print(f"  Colors: {nc},  Valid: {valid},  Steps: {len(steps)}")

    # Axis bounds
    all_x=[p[0] for p in pos.values()]; all_y=[p[1] for p in pos.values()]
    pad=max((max(all_x)-min(all_x)),(max(all_y)-min(all_y)))*0.12+0.5
    XLIM=(min(all_x)-pad,max(all_x)+pad); YLIM=(min(all_y)-pad,max(all_y)+pad)
    ori_edges={tuple(sorted(e)) for e in G_orig.edges()}

    frame_data=list(steps)+[{"victim":None,"outer":set(),"color":None,
                              "color_snap":color,"alive":[],"stack":[],
                              "pfe_nodes":set(),"comp_nodes":set(),
                              "pfes_before":[],"fails_before":{},
                              "fails_after":{},"pfes_after":[],
                              "closure_results":[],"forced":{},"phase":"final"}]

    fig=plt.figure(figsize=(14,8))
    fig.patch.set_facecolor(C_BG)
    ax_g=fig.add_axes([0.01,0.12,0.60,0.86])
    ax_l=fig.add_axes([0.62,0.12,0.37,0.86])
    ax_i=fig.add_axes([0.01,0.00,0.98,0.11])

    def draw(idx):
        f=frame_data[idx]
        is_last=f.get("phase")=="final"
        col=f["color_snap"]; victim=f["victim"]; of=f.get("outer",set())
        alive=f["alive"] if not is_last else list(G_max.nodes())
        pfe_n=f.get("pfe_nodes",set()); comp_n=f.get("comp_nodes",set())
        forced_n=set(f.get("forced",{}).keys())-{victim}
        stack_n=set(f.get("stack",[]))

        # ── Graph ─────────────────────────────────────────────────────────────
        ax_g.clear(); ax_g.set_facecolor(C_BG)
        ax_g.set_aspect("equal"); ax_g.axis("off")
        ax_g.set_xlim(XLIM); ax_g.set_ylim(YLIM)

        ghost=[n for n in G_max.nodes() if n not in alive and n not in stack_n] if not is_last else []
        if ghost:
            gsg=G_max.subgraph(ghost); gps={n:pos[n] for n in ghost}
            nx.draw_networkx_edges(gsg,gps,ax=ax_g,edge_color=C_GHOST,width=0.6,alpha=0.2)
            nx.draw_networkx_nodes(gsg,gps,ax=ax_g,nodelist=ghost,
                                   node_color=C_GHOST,node_size=150,
                                   edgecolors="#BDBDBD",linewidths=0.4)
            nx.draw_networkx_labels(gsg,gps,ax=ax_g,font_size=5,font_color="#BDBDBD")

        draw_n=alive if not is_last else list(G_max.nodes())
        sg2=G_max.subgraph(draw_n); ps={n:pos[n] for n in draw_n}

        for face in tris:
            if all(n in draw_n for n in face):
                in_of=all(n in of for n in face)
                fc="#FFF3E0" if in_of else "#F5F5F5"
                pts=np.array([pos[n] for n in face])
                ax_g.add_patch(Polygon(pts,closed=True,facecolor=fc,edgecolor="none",alpha=0.45))

        e_new=[e for e in sg2.edges() if tuple(sorted(e)) not in ori_edges]
        e_ori=[e for e in sg2.edges() if tuple(sorted(e)) in ori_edges]
        if e_new: nx.draw_networkx_edges(sg2,ps,ax=ax_g,edgelist=e_new,
                                          edge_color="#B2DFDB",width=0.7,alpha=0.35)
        if e_ori: nx.draw_networkx_edges(sg2,ps,ax=ax_g,edgelist=e_ori,
                                          edge_color=C_EDGE,width=1.3,alpha=0.65)

        for n in draw_n:
            c_idx=col.get(n,-1); nc_=PAL.get(c_idx,C_GRAY)
            sz=280; bc="white"; blw=1.0
            if n==victim and not is_last: sz=480; blw=3.0
            elif n in forced_n and not is_last: nc_=C_FORCED; sz=360; blw=2.5
            elif n in pfe_n and not is_last: bc=C_PFE; blw=2.5; sz=320
            elif n in comp_n and not is_last: bc=C_COMP; blw=2.5; sz=320
            elif n in of and not is_last: bc=C_OUT; blw=1.8; sz=300
            if n in stack_n and not is_last: bc=C_STACK; blw=2.0
            nx.draw_networkx_nodes(sg2,ps,ax=ax_g,nodelist=[n],
                                   node_color=nc_,node_size=sz,
                                   edgecolors=bc,linewidths=blw)

        fs=max(5.5,8.5-len(draw_n)*0.1)
        if ps: nx.draw_networkx_labels(sg2,ps,ax=ax_g,font_size=fs,
                                        font_color="white",font_weight="bold")

        # ── Legend ────────────────────────────────────────────────────────────
        ax_l.clear(); ax_l.set_facecolor(C_PANEL); ax_l.axis("off")
        ax_l.text(0.5,0.98,"Bi-Layer Coloring",transform=ax_l.transAxes,
                  ha="center",va="top",fontsize=11,fontweight="bold",color="#1A237E")
        ax_l.text(0.5,0.935,f"De Ita Luna & Marcial-Romero",
                  transform=ax_l.transAxes,ha="center",va="top",
                  fontsize=7.5,color="#546E7A",style="italic")
        ax_l.text(0.5,0.895,graph_name,transform=ax_l.transAxes,
                  ha="center",va="top",fontsize=8.5,color="#546E7A",style="italic")

        y=0.845
        step_n=idx+1
        ax_l.text(0.5,y,"DONE ✓" if is_last else f"Step {step_n}/{len(steps)}",
                  transform=ax_l.transAxes,ha="center",va="top",
                  fontsize=10,fontweight="bold",
                  color="#1B5E20" if is_last else "#E65100")
        y-=0.075

        if not is_last and victim is not None:
            from_stack=f.get("from_stack",False)
            source="[Stack]" if from_stack else "[Outer face]"
            ax_l.text(0.5,y,f"Node {victim}  {source}",transform=ax_l.transAxes,
                      ha="center",va="top",fontsize=9,fontweight="bold",color="#263238")
            y-=0.055
            t=tabu(victim,{k:v for k,v in col.items() if k!=victim},G_max)
            ax_l.text(0.5,y,f"Tabu: {sorted(t)}  →  color {f['color']}",
                      transform=ax_l.transAxes,ha="center",va="top",
                      fontsize=8,color=PAL.get(f['color'],"#333"),fontweight="bold")
            y-=0.055
            if f.get("closure_results"):
                ax_l.text(0.5,y,"Closure evaluation:",transform=ax_l.transAxes,
                          ha="center",va="top",fontsize=7.5,color="#546E7A")
                y-=0.042
                for res in f["closure_results"][:4]:
                    a_,_,cons_,score_=res
                    mark="✓" if cons_ else "✗"
                    sc=str(score_[:3]) if score_ else "inconsistent"
                    ax_l.text(0.5,y,f"  c={a_} {mark} score={sc}",
                              transform=ax_l.transAxes,ha="center",va="top",
                              fontsize=7,color="#27AE60" if cons_ else "#C62828")
                    y-=0.038
            if pfe_n:
                ax_l.text(0.5,y,f"⚠ PFE nodes: {sorted(pfe_n)}",
                          transform=ax_l.transAxes,ha="center",va="top",
                          fontsize=7.5,color=C_PFE,fontweight="bold")
                y-=0.042
            if forced_n:
                ax_l.text(0.5,y,f"→ Forced: {dict((k,v) for k,v in f['forced'].items() if k!=victim)}",
                          transform=ax_l.transAxes,ha="center",va="top",
                          fontsize=7.5,color=C_FORCED,fontweight="bold")
                y-=0.042
            y-=0.01

        ax_l.axhline(y+0.01,xmin=0.05,xmax=0.95,color="#B0BEC5",lw=0.8)
        y-=0.035
        ax_l.text(0.5,y,"Colors used:",transform=ax_l.transAxes,
                  ha="center",va="top",fontsize=8,color="#546E7A")
        y-=0.06
        used=sorted(set(col.values()))
        counts={c:sum(1 for x in col.values() if x==c) for c in used}
        for ci in used[:4]:
            r=FancyBboxPatch((0.06,y-0.036),0.88,0.058,boxstyle="round,pad=0.01",
                             facecolor=PAL.get(ci,"#777"),edgecolor="none",transform=ax_l.transAxes)
            ax_l.add_patch(r)
            ax_l.text(0.5,y,f"{CNAMES.get(ci,'?')}  ×{counts.get(ci,0)}",
                      transform=ax_l.transAxes,ha="center",va="center",
                      fontsize=7.5,color="white",fontweight="bold")
            y-=0.07

        ax_l.axhline(max(y+0.01,0.06),xmin=0.05,xmax=0.95,color="#B0BEC5",lw=0.8)

        # Node type legend
        legend_items=[
            (C_PFE,"white","Potential failed edge"),
            (C_COMP,"white","Complementary tabus"),
            (C_FORCED,"white","Closure-forced node"),
            (C_STACK,"white","Stack (safe) node"),
        ]
        ly=max(y-0.01,0.01)
        for lc,tc,lbl in legend_items:
            if ly<0.02: break
            r=FancyBboxPatch((0.06,ly-0.028),0.88,0.045,boxstyle="round,pad=0.005",
                             facecolor=lc,edgecolor="none",transform=ax_l.transAxes)
            ax_l.add_patch(r)
            ax_l.text(0.5,ly,lbl,transform=ax_l.transAxes,ha="center",va="center",
                      fontsize=6.5,color=tc,fontweight="bold")
            ly-=0.055

        # ── Info bar ──────────────────────────────────────────────────────────
        ax_i.clear(); ax_i.set_facecolor(C_PANEL); ax_i.axis("off")
        if is_last:
            nc_f=len(set(col.values()))
            v_f=all(col.get(u,-1)!=col.get(v,-1) for u,v in G_max.edges() if u in col and v in col)
            title=f"Done  |  {nc_f} colors  |  Valid: {v_f}  |  Bi-Layer (De Ita Luna & Marcial-Romero)"
            body="Closure T((vi,a)): propagate forced colorings → score virtual graph → pick color minimizing pre-failure configs"
        else:
            pfes_b=f.get("pfes_before",[]); fails_b=f.get("fails_before",{})
            title=(f"Step {step_n}/{len(steps)}  |  Node {victim}  →  color {f['color']}  |  "
                   f"PFE: {len(pfes_b)}  |  "
                   f"Fail configs: V={len(fails_b.get('vertices',[]))} E={len(fails_b.get('edges',[]))} F={len(fails_b.get('faces',[]))}")
            body=("Selection: 1st pfe nodes → 2nd complementary tabus → 3rd max degree  |  "
                  "Color: run closure T((vi,a)), score by failure pre-configs, pick minimum")
        ax_i.text(0.012,0.68,title,transform=ax_i.transAxes,
                  fontsize=9,va="center",color="#1A237E",fontweight="bold")
        ax_i.text(0.012,0.22,body,transform=ax_i.transAxes,
                  fontsize=8,va="center",color="#546E7A")

    anim=FuncAnimation(fig,draw,frames=len(frame_data),interval=int(1000/fps),repeat=True)
    if output_path is None: output_path=f"{graph_name}_bilayer.gif"
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    anim.save(output_path,writer=PillowWriter(fps=fps),dpi=110)
    plt.close(fig)
    print(f"  Saved → {output_path}")
    return output_path

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _print_list():
    print(f"\n{'Name':<18} {'Description'}")
    print("─"*60)
    for k,(_,d) in GRAPHS.items(): print(f"  {k:<16} {d}")
    print()

if __name__=="__main__":
    import sys
    if "--list" in sys.argv: _print_list(); sys.exit(0)
    name=sys.argv[1] if len(sys.argv)>1 else "errera"
    out=sys.argv[2] if len(sys.argv)>2 else f"{name}_bilayer.gif"
    render_gif(name, out)
