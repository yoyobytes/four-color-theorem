"""
layer_bipartite_coloring.py
===========================
Layer-Bipartite Coloring: a new algorithm exploiting the bipartite structure
of inner peeling layers to guide 4-coloring without backtracking.

Key observation (Jiménez Muñoz, 2026)
--------------------------------------
When a planar graph is peeled layer by layer (outer face removed at each step),
the inner layers tend to be bipartite graphs (2-colorable internally).
This structure can be exploited to minimize the number of colors used per layer:

  Layer 0 (outer face): always a cycle or triangle.
    - Even cycle → 2-color it with colors {A, B}
    - Odd cycle / triangle → 3-color it with colors {A, B, C}
    - Minimize colors used to leave maximum freedom for inner layers.

  Layer k (inner): often bipartite.
    - If bipartite: identify the two parts (P, Q) of the bipartition.
    - Assign colors to P and Q globally, choosing colors that minimize
      conflicts with cross-layer neighbors.
    - If not bipartite: fall back to greedy with tabu from previous layers.

Formally:
  For each layer L_k:
    1. Compute the subgraph G[L_k].
    2. If G[L_k] is bipartite with parts (P, Q):
       a. For each candidate color pair (c_P, c_Q) from FOUR:
          - Compute total forbidden violations with cross-layer neighbors.
          - Choose (c_P, c_Q) minimizing violations.
       b. Assign c_P to all v in P, c_Q to all v in Q (adjust per-node if needed).
    3. If G[L_k] is not bipartite:
       - Greedy with tabu from already-colored neighbors.

Why this might guarantee 4 colors:
  - Outer face uses ≤ 3 colors (triangle) or ≤ 2 (even cycle).
  - Each inner bipartite layer needs only 2 colors internally.
  - With 4 colors total, there are always 2 colors not used by the outer face
    available for the first inner layer.
  - This propagates inward: each layer's bipartite structure ensures the
    cross-layer forbidden colors leave at least 1 free color per bipartite part.

This is an experimental algorithm. Whether it always produces valid 4-colorings
for all planar graphs is an open question.

Usage
-----
    python layer_bipartite_coloring.py errera
    python layer_bipartite_coloring.py kittell
    python layer_bipartite_coloring.py --list
    python layer_bipartite_coloring.py --compare errera
"""

import math, sys
from pathlib import Path
from itertools import combinations, permutations

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx
import numpy as np

FOUR = {0, 1, 2, 3}
PAL  = {0:"#2196F3", 1:"#4CAF50", 2:"#FF9800", 3:"#E91E63"}
CNAMES = {0:"Blue", 1:"Green", 2:"Orange", 3:"Pink"}

# ── Graph library (shared with other coloring scripts) ────────────────────────
def _circle(nodes, r, a0=0.0):
    nodes=list(nodes); n=len(nodes)
    return {v:(round(r*math.cos(2*math.pi*i/n+a0),4),
               round(r*math.sin(2*math.pi*i/n+a0),4))
            for i,v in enumerate(nodes)}

def _spring(G, scale=7.0, seed=42):
    raw=nx.spring_layout(G, seed=seed, scale=scale)
    return {n:(round(float(x),4),round(float(y),4)) for n,(x,y) in raw.items()}

def graph_errera():
    "Errera (1921) — 17 nodes. χ=4"
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
    "Kittell (1935) — 23 nodes. χ=4"
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
    "Layered 42 — 42 nodes, 5 rings, planar"
    rings=[15,11,8,5,3]; e=[]; offset=0; offsets=[0]
    for r in rings:
        for i in range(r): e.append((offset+i,offset+(i+1)%r))
        offset+=r; offsets.append(offset)
    for k in range(len(rings)-1):
        r_out=rings[k]; r_in=rings[k+1]; off_out=offsets[k]; off_in=offsets[k+1]
        for i in range(r_in):
            j=int(i*r_out/r_in)
            e.append((off_in+i,off_out+j)); e.append((off_in+i,off_out+(j+1)%r_out))
    G=nx.Graph(); G.add_edges_from(e)
    pos={}; radii=[9.5,7.0,5.0,3.0,1.2]
    for k,r in enumerate(rings):
        pos.update(_circle(range(offsets[k],offsets[k+1]),radii[k],math.pi/2))
    return G, pos

def graph_dodecahedron():
    G=nx.convert_node_labels_to_integers(nx.dodecahedral_graph()); return G,_spring(G)
def graph_icosahedron():
    G=nx.convert_node_labels_to_integers(nx.icosahedral_graph()); return G,_spring(G)
def graph_octahedron():
    G=nx.convert_node_labels_to_integers(nx.octahedral_graph()); return G,_spring(G,5)
def graph_wheel_8():
    G=nx.wheel_graph(8); G=nx.convert_node_labels_to_integers(G)
    pos={0:(0,0)}; pos.update(_circle(range(1,8),4,math.pi/2)); return G,pos
def graph_grid_4x4():
    G=nx.convert_node_labels_to_integers(nx.grid_2d_graph(4,4)); return G,_spring(G,6)
def graph_sunflower():
    k=6; e=[]
    for i in range(k):
        a=1+2*i; b=2+2*i; e.extend([(0,a),(0,b),(a,b)]); e.append((b,1+2*((i+1)%k)))
    G=nx.Graph(); G.add_edges_from(e)
    pos={0:(0,0)}; pos.update(_circle(range(1,2*k+1),4,math.pi/2)); return G,pos

GRAPHS = {
    "errera":       (graph_errera,       "Errera (17)        — χ=4"),
    "kittell":      (graph_kittell,      "Kittell (23)       — χ=4"),
    "layered_42":   (graph_layered_42,   "Layered 42 (42)    — 5 rings"),
    "dodecahedron": (graph_dodecahedron, "Dodecahedron (20)  — χ=3"),
    "icosahedron":  (graph_icosahedron,  "Icosahedron (12)   — χ=4"),
    "octahedron":   (graph_octahedron,   "Octahedron (6)     — χ=3"),
    "wheel_8":      (graph_wheel_8,      "Wheel W₈ (8)       — χ=4"),
    "grid_4x4":     (graph_grid_4x4,     "Grid 4×4 (16)      — χ=2"),
    "sunflower":    (graph_sunflower,    "Sunflower (13)     — χ=3"),
}

# ── Planar utilities ──────────────────────────────────────────────────────────
def _outer_face(G):
    if G.number_of_nodes()<=2: return set(G.nodes())
    ok,emb=nx.check_planarity(G)
    if not ok: return set(G.nodes())
    seen=set(); faces=[]
    for v in emb:
        for w in emb.neighbors(v):
            f=tuple(emb.traverse_face(v,w)); k=frozenset(f)
            if k not in seen: seen.add(k); faces.append(list(f))
    try:
        ecc=nx.eccentricity(G)
        return set(max(faces,key=lambda f:sum(ecc[v] for v in set(f))))
    except: return set(G.nodes())

def _maximize(G):
    G2=G.copy()
    cands=sorted([(u,v) for u,v in combinations(G2.nodes(),2) if not G2.has_edge(u,v)],
                 key=lambda e:G2.degree(e[0])+G2.degree(e[1]))
    for u,v in cands:
        G2.add_edge(u,v)
        if not nx.check_planarity(G2)[0]: G2.remove_edge(u,v)
    return G2

def _tutte(G, scale=8.0):
    ok,emb=nx.check_planarity(G); seen=set(); faces=[]
    for v in emb:
        for w in emb.neighbors(v):
            f=tuple(emb.traverse_face(v,w)); k=frozenset(f)
            if k not in seen: seen.add(k); faces.append(list(set(f)))
    ecc=nx.eccentricity(G)
    outer=list(set(max(faces,key=lambda f:sum(ecc[v] for v in set(f)))))
    ko=len(outer)
    pf={v:(scale*math.cos(2*math.pi*i/ko+math.pi/2),
            scale*math.sin(2*math.pi*i/ko+math.pi/2))
        for i,v in enumerate(outer)}
    inn=[v for v in sorted(G.nodes()) if v not in pf]
    if not inn: return pf
    idx={v:i for i,v in enumerate(inn)}; m=len(inn)
    A=np.zeros((m,m)); bx=np.zeros(m); by=np.zeros(m)
    for i,v in enumerate(inn):
        nb=list(G.neighbors(v)); A[i,i]=len(nb)
        for u in nb:
            if u in idx: A[i,idx[u]]=-1
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
# LAYER-BIPARTITE COLORING ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

def _bipartite_parts(G):
    """BFS 2-coloring to get bipartite parts — handles disconnected graphs."""
    col = {}
    for comp in nx.connected_components(G):
        start = min(comp); queue = [start]; col[start] = 0
        while queue:
            v = queue.pop()
            for u in G.neighbors(v):
                if u not in col:
                    col[u] = 1 - col[v]; queue.append(u)
    P = sorted(v for v,c in col.items() if c==0)
    Q = sorted(v for v,c in col.items() if c==1)
    return P, Q


def _color_bipartite_layer(layer_nodes, layer_sg, G_full, color):
    """
    Color a bipartite layer exploiting its bipartite structure.

    Strategy:
    1. Find bipartite parts P and Q.
    2. Find one color for ALL of P: minimum color not forbidden by
       ANY P node's cross-layer neighbors (union of forbidden sets).
       If no such color exists, color P per-node greedily.
    3. Color Q greedily respecting P assignments + cross-layer constraints.

    This minimizes colors: P shares one color, Q shares at most one more,
    keeping the layer within 2 new colors and 4 total.
    """
    P, Q = _bipartite_parts(layer_sg)

    # Step 1: find one color that works for all P nodes simultaneously
    forb_P_union = set()
    for v in P:
        forb_P_union |= {color[u] for u in G_full.neighbors(v) if u in color}

    trial = {}
    conflicts = 0
    avail_P = FOUR - forb_P_union

    if avail_P:
        cp = min(avail_P)
        for v in P:
            trial[v] = cp
    else:
        # No single color for all P — per-node greedy within P
        conflicts = 1
        for v in P:
            cross  = {color[u] for u in G_full.neighbors(v) if u in color}
            within = {trial[u] for u in G_full.neighbors(v) if u in trial}
            avail  = FOUR - cross - within
            trial[v] = min(avail) if avail else min(FOUR - cross, default=0)

    # Step 2: color Q greedily using P assignments + cross-layer
    for v in Q:
        cross  = {color[u] for u in G_full.neighbors(v) if u in color}
        within = {trial[u] for u in G_full.neighbors(v) if u in trial}
        avail  = FOUR - cross - within
        if avail:
            trial[v] = min(avail)
        else:
            trial[v] = min(FOUR - cross) if (FOUR - cross) else 0
            conflicts += 1

    # Verify no internal edge conflict
    valid = all(trial.get(u) != trial.get(v) for u, v in layer_sg.edges())
    if not valid:
        return None, float('inf')

    return trial, conflicts


def _color_layer_greedy(layer_nodes, G_full, color):
    """Fallback: greedy coloring for non-bipartite layers."""
    result = {}
    for v in layer_nodes:
        forb = {color[u] for u in G_full.neighbors(v) if u in color}
        forb |= {result[u] for u in G_full.neighbors(v) if u in result}
        c = 0
        while c in forb: c += 1
        result[v] = c
    return result


def layer_bipartite_coloring(G_orig):
    """
    Main algorithm: color the graph layer by layer exploiting bipartite structure.

    Returns (color, steps) where each step dict contains:
        layer_idx     — which layer this is
        layer_nodes   — nodes in this layer
        is_bipartite  — whether the layer subgraph is bipartite
        parts         — (P, Q) bipartite parts if bipartite, else None
        part_colors   — (color_P, color_Q) chosen for the parts
        color_snap    — full coloring after this layer
        conflicts     — number of nodes that deviated from part color
    """
    sg = G_orig.copy()
    color = {}
    steps = []
    layer_idx = 0

    while sg.number_of_nodes() > 0:
        of = _outer_face(sg)
        layer = sorted(of)
        layer_sg = G_orig.subgraph(layer)

        is_bip = nx.is_bipartite(layer_sg)

        if is_bip and len(layer) > 1:
            layer_col, conflicts = _color_bipartite_layer(
                layer, layer_sg, G_orig, color)
            if layer_col is None:
                # bipartite optimization failed — use greedy
                layer_col = _color_layer_greedy(layer, G_orig, color)
                conflicts = -1  # signal fallback
            P, Q = _bipartite_parts(layer_sg)
        else:
            layer_col = _color_layer_greedy(layer, G_orig, color)
            conflicts = 0
            P, Q = layer, []

        # Detect dominant part colors (most common in each part)
        p_colors = [layer_col.get(v) for v in P if layer_col.get(v) is not None]
        q_colors = [layer_col.get(v) for v in Q if layer_col.get(v) is not None]
        cp = max(set(p_colors), key=p_colors.count) if p_colors else None
        cq = max(set(q_colors), key=q_colors.count) if q_colors else None

        color.update(layer_col)

        steps.append({
            "layer_idx":    layer_idx,
            "layer_nodes":  layer,
            "is_bipartite": is_bip,
            "P":            P,
            "Q":            Q,
            "cp":           cp,
            "cq":           cq,
            "layer_col":    dict(layer_col),
            "color_snap":   dict(color),
            "conflicts":    conflicts,
            "alive_before": list(sg.nodes()),
        })

        for v in list(of):
            if v in sg: sg.remove_node(v)
        layer_idx += 1

    return color, steps

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

C_BG="#FAFAFA"; C_PANEL="#ECEFF1"; C_EDGE="#607D8B"; C_GHOST="#E0E0E0"
C_P="#1565C0"; C_Q="#AD1457"  # bipartite part colors for borders

# Layer background colors
LAYER_BG=["#FFF3E0","#E8F5E9","#E3F2FD","#FCE4EC","#F3E5F5","#E0F2F1"]

def render_gif(graph_name=None, output_path=None, fps=0.7,
               G_override=None, pos_override=None, name_override=None):
    if G_override is not None:
        G_orig, pos_orig = G_override, pos_override
        graph_name = name_override or "custom"
    elif graph_name in GRAPHS:
        fn, _ = GRAPHS[graph_name]
        G_orig, pos_orig = fn()
    else:
        raise KeyError(f"Unknown graph '{graph_name}'")

    print(f"\n  {graph_name}: {G_orig.number_of_nodes()}V {G_orig.number_of_edges()}E")
    if not nx.check_planarity(G_orig)[0]:
        print("  ✗ Not planar."); return

    print("  Computing layout...")
    G_max = _maximize(G_orig)
    pos   = _tutte(G_max)
    tris  = _tri_faces(G_max)

    print("  Running Layer-Bipartite algorithm...")
    color, steps = layer_bipartite_coloring(G_max)

    nc    = len(set(color.values()))
    valid = all(color.get(u,-1)!=color.get(v,-1) for u,v in G_max.edges()
                if u in color and v in color)
    print(f"  Colors: {nc},  Valid: {valid},  Layers: {len(steps)}")

    # Axis bounds
    all_x=[p[0] for p in pos.values()]; all_y=[p[1] for p in pos.values()]
    pad=max((max(all_x)-min(all_x)),(max(all_y)-min(all_y)))*0.12+0.5
    XLIM=(min(all_x)-pad,max(all_x)+pad); YLIM=(min(all_y)-pad,max(all_y)+pad)
    ori_edges={tuple(sorted(e)) for e in G_orig.edges()}

    # Build frame list: one frame per node assignment within each layer
    frames=[]
    cumulative_col={}
    for s in steps:
        layer=s["layer_nodes"]; P=set(s["P"]); Q=set(s["Q"])
        layer_col=s["layer_col"]; is_bip=s["is_bipartite"]
        alive=s["alive_before"]

        for node in layer:
            cumulative_col[node]=layer_col[node]
            frames.append({
                "layer_idx":  s["layer_idx"],
                "is_bip":     is_bip,
                "P": P, "Q": Q,
                "cp": s["cp"], "cq": s["cq"],
                "node":       node,
                "color":      layer_col[node],
                "color_snap": dict(cumulative_col),
                "layer_nodes": layer,
                "alive":      alive,
                "conflicts":  s["conflicts"],
            })

    frames.append({"phase":"final","color_snap":color,
                   "alive":list(G_max.nodes()),"layer_idx":len(steps),
                   "is_bip":False,"P":[],"Q":[],"cp":None,"cq":None,
                   "node":None,"color":None,"layer_nodes":[],"conflicts":0})

    fig=plt.figure(figsize=(14,8))
    fig.patch.set_facecolor(C_BG)
    ax_g=fig.add_axes([0.01,0.12,0.60,0.86])
    ax_l=fig.add_axes([0.62,0.12,0.37,0.86])
    ax_i=fig.add_axes([0.01,0.00,0.98,0.11])

    def draw(idx):
        f=frames[idx]
        is_last=f.get("phase")=="final"
        col=f["color_snap"]; alive=f["alive"]
        layer_n=set(f["layer_nodes"]); P=set(f["P"]); Q=set(f["Q"])
        node=f["node"]; is_bip=f["is_bip"]
        cp=f["cp"]; cq=f["cq"]; layer_idx=f["layer_idx"]

        # ── Graph ─────────────────────────────────────────────────────────────
        ax_g.clear(); ax_g.set_facecolor(C_BG)
        ax_g.set_aspect("equal"); ax_g.axis("off")
        ax_g.set_xlim(XLIM); ax_g.set_ylim(YLIM)

        # Ghost
        ghost=[n for n in G_max.nodes() if n not in alive] if not is_last else []
        if ghost:
            gsg=G_max.subgraph(ghost); gps={n:pos[n] for n in ghost}
            nx.draw_networkx_edges(gsg,gps,ax=ax_g,edge_color=C_GHOST,width=0.6,alpha=0.2)
            nx.draw_networkx_nodes(gsg,gps,ax=ax_g,nodelist=ghost,
                                   node_color=C_GHOST,node_size=150,
                                   edgecolors="#BDBDBD",linewidths=0.4)
            nx.draw_networkx_labels(gsg,gps,ax=ax_g,font_size=5,font_color="#BDBDBD")

        draw_n=alive if not is_last else list(G_max.nodes())
        sg2=G_max.subgraph(draw_n); ps={n:pos[n] for n in draw_n}

        # Layer background highlight
        lbg=LAYER_BG[layer_idx%len(LAYER_BG)] if not is_last else "#F5F5F5"
        for face in tris:
            if all(n in draw_n for n in face):
                in_layer=all(n in layer_n for n in face)
                fc=lbg if in_layer else "#F5F5F5"
                pts=np.array([pos[n] for n in face])
                ax_g.add_patch(Polygon(pts,closed=True,facecolor=fc,edgecolor="none",alpha=0.5))

        # Edges
        e_new=[e for e in sg2.edges() if tuple(sorted(e)) not in ori_edges]
        e_ori=[e for e in sg2.edges() if tuple(sorted(e)) in ori_edges]
        if e_new: nx.draw_networkx_edges(sg2,ps,ax=ax_g,edgelist=e_new,
                                          edge_color="#B2DFDB",width=0.7,alpha=0.35)
        if e_ori: nx.draw_networkx_edges(sg2,ps,ax=ax_g,edgelist=e_ori,
                                          edge_color=C_EDGE,width=1.3,alpha=0.65)

        # Nodes
        for n in draw_n:
            c_idx=col.get(n,-1); nc_=PAL.get(c_idx,"#90A4AE")
            sz=280; blw=1.0
            # Border color encodes bipartite part
            if n in P and not is_last: bc=C_P; blw=2.2
            elif n in Q and not is_last: bc=C_Q; blw=2.2
            else: bc="white"
            if n==node and not is_last: sz=460; blw=3.5
            nx.draw_networkx_nodes(sg2,ps,ax=ax_g,nodelist=[n],
                                   node_color=nc_,node_size=sz,
                                   edgecolors=bc,linewidths=blw)

        fs=max(5.5,8.5-len(draw_n)*0.1)
        if ps: nx.draw_networkx_labels(sg2,ps,ax=ax_g,font_size=fs,
                                        font_color="white",font_weight="bold")

        # ── Legend ────────────────────────────────────────────────────────────
        ax_l.clear(); ax_l.set_facecolor(C_PANEL); ax_l.axis("off")
        ax_l.text(0.5,0.98,"Layer-Bipartite Coloring",transform=ax_l.transAxes,
                  ha="center",va="top",fontsize=11,fontweight="bold",color="#1A237E")
        ax_l.text(0.5,0.935,graph_name,transform=ax_l.transAxes,
                  ha="center",va="top",fontsize=8.5,color="#546E7A",style="italic")

        y=0.875
        if is_last:
            nc_f=len(set(col.values()))
            ax_l.text(0.5,y,f"Done ✓  {nc_f} colors",transform=ax_l.transAxes,
                      ha="center",va="top",fontsize=12,fontweight="bold",color="#1B5E20")
            y-=0.08
        else:
            ax_l.text(0.5,y,f"Layer {layer_idx}",transform=ax_l.transAxes,
                      ha="center",va="top",fontsize=11,fontweight="bold",color="#E65100")
            y-=0.065
            bip_str="Bipartite ✓" if is_bip else "Not bipartite"
            bip_col="#1B5E20" if is_bip else "#C62828"
            ax_l.text(0.5,y,bip_str,transform=ax_l.transAxes,
                      ha="center",va="top",fontsize=9,color=bip_col,fontweight="bold")
            y-=0.065
            if is_bip and cp is not None:
                for part,pcol,cc in [("Part P",C_P,cp),("Part Q",C_Q,cq)]:
                    r=FancyBboxPatch((0.06,y-0.038),0.38,0.062,
                                     boxstyle="round,pad=0.01",
                                     facecolor=pcol,edgecolor="none",transform=ax_l.transAxes)
                    ax_l.add_patch(r)
                    ax_l.text(0.25,y,part,transform=ax_l.transAxes,
                              ha="center",va="center",fontsize=7.5,color="white",fontweight="bold")
                    r2=FancyBboxPatch((0.54,y-0.038),0.40,0.062,
                                      boxstyle="round,pad=0.01",
                                      facecolor=PAL.get(cc,"#777"),edgecolor="none",
                                      transform=ax_l.transAxes)
                    ax_l.add_patch(r2)
                    ax_l.text(0.74,y,f"color {cc} ({CNAMES.get(cc,'?')})",
                              transform=ax_l.transAxes,ha="center",va="center",
                              fontsize=7.5,color="white",fontweight="bold")
                    y-=0.075
            if node is not None:
                ax_l.text(0.5,y,f"→ node {node} = {f['color']} ({CNAMES.get(f['color'],'?')})",
                          transform=ax_l.transAxes,ha="center",va="top",
                          fontsize=8.5,fontweight="bold",color=PAL.get(f['color'],"#333"))
                y-=0.065

        ax_l.axhline(y,xmin=0.05,xmax=0.95,color="#B0BEC5",lw=0.8)
        y-=0.04
        ax_l.text(0.5,y,"Colors used:",transform=ax_l.transAxes,
                  ha="center",va="top",fontsize=8,color="#546E7A")
        y-=0.065
        used=sorted(set(col.values()))
        counts={c:sum(1 for x in col.values() if x==c) for c in used}
        for ci in used[:4]:
            r=FancyBboxPatch((0.06,y-0.036),0.88,0.058,
                              boxstyle="round,pad=0.01",facecolor=PAL.get(ci,"#777"),
                              edgecolor="none",transform=ax_l.transAxes)
            ax_l.add_patch(r)
            ax_l.text(0.5,y,f"{CNAMES.get(ci,'?')}  ×{counts.get(ci,0)}",
                      transform=ax_l.transAxes,ha="center",va="center",
                      fontsize=7.5,color="white",fontweight="bold")
            y-=0.072

        # Bipartite legend
        ax_l.axhline(max(y,0.09),xmin=0.05,xmax=0.95,color="#B0BEC5",lw=0.8)
        legy=max(y-0.02,0.01)
        for bc,lbl in [(C_P,"Part P (border)"),(C_Q,"Part Q (border)")]:
            if legy<0.02: break
            r=FancyBboxPatch((0.06,legy-0.025),0.88,0.042,
                              boxstyle="round,pad=0.005",facecolor=bc,edgecolor="none",
                              transform=ax_l.transAxes)
            ax_l.add_patch(r)
            ax_l.text(0.5,legy,lbl,transform=ax_l.transAxes,ha="center",va="center",
                      fontsize=6.5,color="white",fontweight="bold")
            legy-=0.05

        # ── Info bar ──────────────────────────────────────────────────────────
        ax_i.clear(); ax_i.set_facecolor(C_PANEL); ax_i.axis("off")
        if is_last:
            nc_f=len(set(col.values()))
            v_f=all(col.get(u,-1)!=col.get(v,-1) for u,v in G_max.edges() if u in col and v in col)
            title=f"Done  |  {nc_f} colors  |  Valid: {v_f}  |  Layer-Bipartite Coloring"
            body="Each layer colored using its bipartite structure: assign one color to each part, minimize cross-layer conflicts"
        else:
            title=(f"Layer {layer_idx}  |  {'Bipartite' if is_bip else 'Non-bipartite'}  |  "
                   f"Node {node} → color {f['color']} ({CNAMES.get(f['color'],'?')})  |  "
                   f"Step {idx+1}/{len(frames)-1}")
            body=(f"Layer nodes: {sorted(layer_n)}  |  "
                  f"P (blue border): {sorted(P)}  Q (pink border): {sorted(Q)}")
        ax_i.text(0.012,0.68,title,transform=ax_i.transAxes,
                  fontsize=9,va="center",color="#1A237E",fontweight="bold")
        ax_i.text(0.012,0.22,body,transform=ax_i.transAxes,
                  fontsize=8,va="center",color="#546E7A")

    anim=FuncAnimation(fig,draw,frames=len(frames),interval=int(1000/fps),repeat=True)
    if output_path is None: output_path=f"{graph_name}_layer_bipartite.gif"
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    anim.save(output_path,writer=PillowWriter(fps=fps),dpi=110)
    plt.close(fig)
    print(f"  Saved → {output_path}")
    return output_path

# ── Comparison function ───────────────────────────────────────────────────────
def compare_all(graph_name):
    """Compare tabu, bilayer, and layer-bipartite on the same graph."""
    if graph_name not in GRAPHS:
        print(f"Unknown graph '{graph_name}'"); return
    fn, desc = GRAPHS[graph_name]
    G, pos = fn()
    G_max = _maximize(G)

    print(f"\n{'='*55}")
    print(f"  Comparison: {desc}")
    print(f"{'='*55}")

    # Layer-Bipartite
    lb_color, lb_steps = layer_bipartite_coloring(G_max)
    lb_nc = len(set(lb_color.values()))
    lb_valid = all(lb_color.get(u,-1)!=lb_color.get(v,-1) for u,v in G_max.edges() if u in lb_color and v in lb_color)

    # Tabu (node-by-node)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import tabu_coloring as T
        _, _, _, tabu_steps, tabu_color = T.build_steps(G, pos)
        tab_nc = len(set(tabu_color.values()))
        tab_valid = all(tabu_color[u]!=tabu_color[v] for u,v in G_max.edges() if u in tabu_color and v in tabu_color)
    except Exception as e:
        tab_nc, tab_valid, tabu_steps = "N/A", "N/A", []

    print(f"\n  {'Method':<28} {'Colors':>7} {'Valid':>6} {'Steps':>7}")
    print(f"  {'-'*50}")
    print(f"  {'Tabu (node-by-node)':<28} {str(tab_nc):>7} {str(tab_valid):>6} {len(tabu_steps):>7}")
    print(f"  {'Layer-Bipartite':<28} {lb_nc:>7} {str(lb_valid):>6} {len(lb_steps):>7}")
    print()

    # Layer analysis
    print(f"  Layer structure:")
    for s in lb_steps:
        bip_str="bipartite" if s["is_bipartite"] else "NON-bipartite"
        print(f"    Layer {s['layer_idx']}: {sorted(s['layer_nodes'])}  [{bip_str}]")

# ── Entry point ───────────────────────────────────────────────────────────────
def _print_list():
    print(f"\n{'Name':<18} {'Description'}")
    print("─"*60)
    for k,(_,d) in GRAPHS.items(): print(f"  {k:<16} {d}")
    print()

if __name__=="__main__":
    if "--list" in sys.argv: _print_list(); sys.exit(0)
    if "--compare" in sys.argv:
        name=sys.argv[sys.argv.index("--compare")+1] if len(sys.argv)>2 else "errera"
        compare_all(name); sys.exit(0)
    name=sys.argv[1] if len(sys.argv)>1 else "errera"
    out=sys.argv[2] if len(sys.argv)>2 else f"{name}_layer_bipartite.gif"
    render_gif(name, out)
