"""
tabu_coloring.py
================
Node-by-node tabu coloring visualizer for planar graphs.

Algorithm
---------
At each step:
  1. Compute the current outer face of the surviving subgraph.
  2. Among those nodes, pick the one with the fewest tabu colors
     (ties broken by node ID — most constrained first).
  3. Assign the minimum color not in its tabu set.
     Tabu(v) = colors of already-colored direct neighbors.
  4. Remove that node. Repeat.

No backtracking. The outer face shrinks inward one node at a time.

Usage
-----
    python tabu_coloring.py                  # interactive menu
    python tabu_coloring.py errera           # run named example
    python tabu_coloring.py --list           # list all graphs
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
from shapely.geometry import LineString, Point

# ══════════════════════════════════════════════════════════════════════════════
# GRAPH LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

def _circle(nodes, r, a0=0.0):
    n = len(list(nodes)); nodes = list(nodes)
    return {v: (round(r*math.cos(2*math.pi*i/n+a0),4),
                round(r*math.sin(2*math.pi*i/n+a0),4))
            for i,v in enumerate(nodes)}

def _spring(G, scale=7.0, seed=42):
    raw = nx.spring_layout(G, seed=seed, scale=scale)
    return {n:(round(float(x),4),round(float(y),4)) for n,(x,y) in raw.items()}

# ── Kempe counterexamples ──────────────────────────────────────────────────────
def graph_errera():
    "Errera (1921) — 17 nodes, 45 edges. 1st Kempe counterexample. χ=4"
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

# ── Classic polyhedra ──────────────────────────────────────────────────────────
def graph_dodecahedron():
    "Dodecahedron — 20 nodes, 30 edges. 3-regular, χ=3"
    G=nx.convert_node_labels_to_integers(nx.dodecahedral_graph())
    return G, _spring(G)

def graph_icosahedron():
    "Icosahedron — 12 nodes, 30 edges. 5-regular, χ=4"
    G=nx.convert_node_labels_to_integers(nx.icosahedral_graph())
    return G, _spring(G)

def graph_octahedron():
    "Octahedron — 6 nodes, 12 edges. 4-regular, χ=3"
    G=nx.convert_node_labels_to_integers(nx.octahedral_graph())
    return G, _spring(G, scale=5)

def graph_cuboctahedron():
    "Cuboctahedron — 12 nodes. 4-regular, χ=3"
    G=nx.convert_node_labels_to_integers(nx.cuboctahedral_graph())
    return G, _spring(G)

# ── Wheels and grids ───────────────────────────────────────────────────────────
def graph_wheel_8():
    "Wheel W₈ — 8 nodes. Hub + 7-cycle. χ=4 (odd rim)"
    G=nx.wheel_graph(8); G=nx.convert_node_labels_to_integers(G)
    pos={0:(0.0,0.0)}; pos.update(_circle(range(1,8),4.0,math.pi/2))
    return G, pos

def graph_wheel_9():
    "Wheel W₉ — 9 nodes. Hub + 8-cycle. χ=3 (even rim)"
    G=nx.wheel_graph(9); G=nx.convert_node_labels_to_integers(G)
    pos={0:(0.0,0.0)}; pos.update(_circle(range(1,9),4.0,math.pi/2))
    return G, pos

def graph_grid_4x4():
    "4×4 grid — 16 nodes. Simple planar, χ=2"
    G=nx.convert_node_labels_to_integers(nx.grid_2d_graph(4,4))
    return G, _spring(G, scale=6)

def graph_grid_3x5():
    "3×5 grid — 15 nodes. Simple planar, χ=2"
    G=nx.convert_node_labels_to_integers(nx.grid_2d_graph(3,5))
    return G, _spring(G, scale=6)

def graph_triangular_grid():
    "Triangular grid — 12 nodes. Dense planar, χ=3"
    rows=[[0,1,2,3],[4,5,6,7],[8,9,10,11]]; e=[]
    for row in rows:
        for i in range(len(row)-1): e.append((row[i],row[i+1]))
    for i in range(len(rows)-1):
        for j in range(len(rows[i])):
            e.append((rows[i][j],rows[i+1][j]))
        for j in range(len(rows[i])-1):
            e.append((rows[i][j],rows[i+1][j+1]))
    G=nx.Graph(); G.add_edges_from(e)
    pos={}
    for i,row in enumerate(rows):
        for j,v in enumerate(row): pos[v]=(j*2.0+i*0.5,-i*1.8)
    return G, pos

# ── Layered / concentric ───────────────────────────────────────────────────────
def graph_layered_42():
    "Layered 42 — 42 nodes, 5 concentric rings, planar (rings=[15,11,8,5,3])"
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

def graph_sunflower():
    "Sunflower — 13 nodes. 6 petals sharing center. χ=3"
    k=6; e=[]
    for i in range(k):
        a=1+2*i; b=2+2*i
        e.extend([(0,a),(0,b),(a,b)])
        e.append((b,1+2*((i+1)%k)))
    G=nx.Graph(); G.add_edges_from(e)
    pos={0:(0.0,0.0)}; pos.update(_circle(range(1,2*k+1),4.0,math.pi/2))
    return G, pos

def graph_double_wheel():
    "Double wheel — 11 nodes. Two concentric pentagons + hub. χ=4"
    e=[(i,(i+1)%5) for i in range(5)]
    e+=[(5+i,(5+i+1)%5+5) for i in range(5)]
    e+=[(i,5+i) for i in range(5)]
    e+=[(10,i) for i in range(10)]
    G=nx.Graph(); G.add_edges_from(e)
    pos={10:(0,0)}
    pos.update(_circle(range(5),3.0,math.pi/2))
    pos.update(_circle(range(5,10),6.0,math.pi/2))
    return G, pos

def graph_book_5():
    "Book B₅ — 7 nodes. 5 triangles sharing a common edge. χ=3"
    n=5; e=[(0,1)]
    for i in range(n): v=2+i; e.extend([(0,v),(1,v)])
    G=nx.Graph(); G.add_edges_from(e)
    pos={0:(-2.0,0.0),1:(2.0,0.0)}
    for i in range(n):
        a=2*math.pi*i/n+math.pi/2
        pos[2+i]=(round(5*math.cos(a),4),round(5*math.sin(a),4))
    return G, pos

def graph_stacked_triangles():
    "Stacked triangles — 10 nodes. Chain of 5 triangles. χ=3"
    e=[(0,1),(1,2),(2,0),(1,3),(3,4),(4,1),(3,5),(5,6),(6,3),
       (5,7),(7,8),(8,5),(7,9),(9,0),(0,7)]
    G=nx.Graph(); G.add_edges_from(e)
    return G, _spring(G, scale=5)

def graph_petersen_planar():
    "Planar Petersen-like — 10 nodes. χ=3"
    e=[(i,(i+1)%5) for i in range(5)]
    e+=[(i,5+i) for i in range(5)]
    e+=[(5,7),(6,8),(7,9),(8,5),(9,6)]
    G=nx.Graph(); G.add_edges_from(e)
    pos={}
    pos.update(_circle(range(5),4.0,math.pi/2))
    pos.update(_circle(range(5,10),2.0,math.pi/2))
    return G, pos

# ── Named graphs ───────────────────────────────────────────────────────────────
def graph_heawood():
    "Heawood — 14 nodes. 3-regular, χ=2"
    G=nx.convert_node_labels_to_integers(nx.heawood_graph())
    return G, _spring(G)

def graph_franklin():
    "Franklin — 12 nodes. 3-regular, χ=2"
    G=nx.convert_node_labels_to_integers(nx.franklin_graph())
    return G, _spring(G)

def graph_chvatal():
    "Chvátal — 12 nodes. Triangle-free, 4-regular, χ=4"
    G=nx.convert_node_labels_to_integers(nx.chvatal_graph())
    return G, _spring(G)

def graph_pappus():
    "Pappus — 18 nodes. 3-regular, bipartite, χ=2"
    G=nx.convert_node_labels_to_integers(nx.pappus_graph())
    return G, _spring(G)

def graph_desargues():
    "Desargues — 20 nodes. 3-regular, χ=2"
    G=nx.convert_node_labels_to_integers(nx.desargues_graph())
    return G, _spring(G)

# ── Random triangulations ──────────────────────────────────────────────────────
def graph_random_20():
    "Random triangulation — 20 nodes. Delaunay"
    import random; random.seed(7)
    pts=[(random.uniform(-1,1),random.uniform(-1,1)) for _ in range(20)]
    try:
        from scipy.spatial import Delaunay
        tri=Delaunay(pts); e=set()
        for s in tri.simplices:
            for i in range(3): a,b=s[i],s[(i+1)%3]; e.add((min(a,b),max(a,b)))
        G=nx.Graph(); G.add_edges_from(e)
        pos={i:(round(pts[i][0]*7,4),round(pts[i][1]*7,4)) for i in range(20)}
        return G, pos
    except:
        G=nx.convert_node_labels_to_integers(nx.triangular_lattice_graph(3,3))
        return G, _spring(G)

def graph_random_30():
    "Random triangulation — 30 nodes. Delaunay"
    import random; random.seed(13)
    pts=[(random.uniform(-1,1),random.uniform(-1,1)) for _ in range(30)]
    try:
        from scipy.spatial import Delaunay
        tri=Delaunay(pts); e=set()
        for s in tri.simplices:
            for i in range(3): a,b=s[i],s[(i+1)%3]; e.add((min(a,b),max(a,b)))
        G=nx.Graph(); G.add_edges_from(e)
        pos={i:(round(pts[i][0]*8,4),round(pts[i][1]*8,4)) for i in range(30)}
        return G, pos
    except:
        G=nx.convert_node_labels_to_integers(nx.triangular_lattice_graph(4,4))
        return G, _spring(G)

def graph_random_15():
    "Random triangulation — 15 nodes. Delaunay"
    import random; random.seed(99)
    pts=[(random.uniform(-1,1),random.uniform(-1,1)) for _ in range(15)]
    try:
        from scipy.spatial import Delaunay
        tri=Delaunay(pts); e=set()
        for s in tri.simplices:
            for i in range(3): a,b=s[i],s[(i+1)%3]; e.add((min(a,b),max(a,b)))
        G=nx.Graph(); G.add_edges_from(e)
        pos={i:(round(pts[i][0]*7,4),round(pts[i][1]*7,4)) for i in range(15)}
        return G, pos
    except:
        G=nx.convert_node_labels_to_integers(nx.triangular_lattice_graph(2,4))
        return G, _spring(G)

# ── Registry ───────────────────────────────────────────────────────────────────
GRAPHS = {
    "errera":           (graph_errera,          "Errera (17)            — 1st Kempe counterexample, χ=4"),
    "kittell":          (graph_kittell,         "Kittell (23)           — 2nd Kempe counterexample, χ=4"),
    "dodecahedron":     (graph_dodecahedron,    "Dodecahedron (20)      — 3-regular, χ=3"),
    "icosahedron":      (graph_icosahedron,     "Icosahedron (12)       — 5-regular, χ=4"),
    "octahedron":       (graph_octahedron,      "Octahedron (6)         — 4-regular, χ=3"),
    "cuboctahedron":    (graph_cuboctahedron,   "Cuboctahedron (12)     — 4-regular, χ=3"),
    "wheel_8":          (graph_wheel_8,         "Wheel W₈ (8)           — odd rim, χ=4"),
    "wheel_9":          (graph_wheel_9,         "Wheel W₉ (9)           — even rim, χ=3"),
    "grid_4x4":         (graph_grid_4x4,        "Grid 4×4 (16)          — χ=2"),
    "grid_3x5":         (graph_grid_3x5,        "Grid 3×5 (15)          — χ=2"),
    "triangular_grid":  (graph_triangular_grid, "Triangular grid (12)   — χ=3"),
    "layered_42":       (graph_layered_42,      "Layered 42 (42)        — 4 concentric rings"),
    "sunflower":        (graph_sunflower,       "Sunflower (13)         — 6 petals, χ=3"),
    "double_wheel":     (graph_double_wheel,    "Double wheel (11)      — χ=4"),
    "petersen_planar":  (graph_petersen_planar, "Planar Petersen (10)   — χ=3"),
    "book_5":           (graph_book_5,          "Book B₅ (7)            — 5 triangles, χ=3"),
    "stacked_tri":      (graph_stacked_triangles,"Stacked triangles (10) — χ=3"),
    "heawood":          (graph_heawood,         "Heawood (14)           — 3-regular, χ=2"),
    "franklin":         (graph_franklin,        "Franklin (12)          — 3-regular, χ=2"),
    "chvatal":          (graph_chvatal,         "Chvátal (12)           — triangle-free, χ=4"),
    "pappus":           (graph_pappus,          "Pappus (18)            — 3-regular, χ=2"),
    "desargues":        (graph_desargues,       "Desargues (20)         — 3-regular, χ=2"),
    "random_15":        (graph_random_15,       "Random triangulation (15)"),
    "random_20":        (graph_random_20,       "Random triangulation (20)"),
    "random_30":        (graph_random_30,       "Random triangulation (30)"),
}

# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

def _outer_face(G2):
    if G2.number_of_nodes()<=2: return set(G2.nodes())
    ok,emb=nx.check_planarity(G2)
    if not ok: return set(G2.nodes())
    seen=set(); faces=[]
    for v in emb:
        for w in emb.neighbors(v):
            f=tuple(emb.traverse_face(v,w)); k=frozenset(f)
            if k not in seen: seen.add(k); faces.append(list(f))
    try:
        ecc=nx.eccentricity(G2)
        return set(max(faces,key=lambda f:sum(ecc[v] for v in set(f))))
    except:
        return set(G2.nodes())

def _maximize(G, pos=None):
    """Add edges using combinatorial planarity check (correct, no geometry needed)."""
    G2=G.copy()
    cands=sorted([(u,v) for u,v in combinations(G2.nodes(),2) if not G2.has_edge(u,v)],
                 key=lambda e:G2.degree(e[0])+G2.degree(e[1]))
    for u,v in cands:
        G2.add_edge(u,v)
        if not nx.check_planarity(G2)[0]:
            G2.remove_edge(u,v)
    return G2

def _tutte(G, scale=8.0):
    ok,emb=nx.check_planarity(G)
    seen=set(); faces=[]
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
    try:
        xc=np.linalg.solve(A,bx); yc=np.linalg.solve(A,by)
    except:
        return {**pf,**{v:(0.0,0.0) for v in inn}}
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

def build_steps(G_orig, pos_orig):
    if not nx.check_planarity(G_orig)[0]:
        raise ValueError("Graph is not planar.")
    G_max=_maximize(G_orig,pos_orig)
    pos=_tutte(G_max)
    tris=_tri_faces(G_max)
    sg=G_max.copy(); color={}; steps=[]
    while sg.number_of_nodes()>0:
        of=_outer_face(sg)
        def ts(v): return len({color[nb] for nb in G_max.neighbors(v) if nb in color})
        victim=min(of,key=lambda v:(ts(v),v))
        tabu=sorted({color[nb] for nb in G_max.neighbors(victim) if nb in color})
        c=0
        while c in tabu: c+=1
        color[victim]=c
        steps.append({"victim":victim,"outer":set(of),"tabu":tabu,
                      "color":c,"color_snap":dict(color),"alive":sorted(sg.nodes())})
        sg.remove_node(victim)
    return G_max, pos, tris, steps, color

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

PAL={0:"#2196F3",1:"#4CAF50",2:"#FF9800",3:"#E91E63",4:"#9C27B0",5:"#009688"}
CNAMES={0:"Blue",1:"Green",2:"Orange",3:"Pink",4:"Purple",5:"Teal"}
C_OUT="#FB8C00"; C_BG="#FAFAFA"; C_PANEL="#ECEFF1"
C_GRAY="#90A4AE"; C_EDGE="#607D8B"; C_GHOST="#E0E0E0"

def render_gif(graph_name=None, output_path=None, fps=0.9,
               G_override=None, pos_override=None, name_override=None):
    if G_override is not None:
        G_orig, pos_orig = G_override, pos_override
        graph_name = name_override or "custom"
        desc = f"{graph_name} ({G_orig.number_of_nodes()} nodes)"
    elif graph_name and graph_name in GRAPHS:
        fn, desc = GRAPHS[graph_name]
        G_orig, pos_orig = fn()
    else:
        raise KeyError(f"Unknown graph '{graph_name}'. Use --list.")
    print(f"\n  {graph_name}: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges")
    if not nx.check_planarity(G_orig)[0]:
        print("  ✗ Not planar."); return
    print("  Running tabu algorithm...")
    G_max, pos, tris, steps, final_color = build_steps(G_orig, pos_orig)
    nc=len(set(final_color.values()))
    valid=all(final_color[u]!=final_color[v] for u,v in G_max.edges())
    print(f"  Colors: {nc},  Valid: {valid},  Steps: {len(steps)}")

    # Fixed axis bounds
    all_x=[p[0] for p in pos.values()]; all_y=[p[1] for p in pos.values()]
    pad=max((max(all_x)-min(all_x)),(max(all_y)-min(all_y)))*0.12+0.5
    XLIM=(min(all_x)-pad,max(all_x)+pad)
    YLIM=(min(all_y)-pad,max(all_y)+pad)
    ori_edges={tuple(sorted(e)) for e in G_orig.edges()}

    frame_data=list(steps)+[{"victim":None,"outer":set(),"tabu":[],"color":-1,
                              "color_snap":final_color,"alive":[],"phase":"final"}]

    fig=plt.figure(figsize=(13,8))
    fig.patch.set_facecolor(C_BG)
    ax_g=fig.add_axes([0.01,0.12,0.65,0.86])
    ax_l=fig.add_axes([0.67,0.12,0.31,0.86])
    ax_i=fig.add_axes([0.01,0.00,0.98,0.11])

    def draw(idx):
        f=frame_data[idx]
        is_last=f.get("phase")=="final"
        col=f["color_snap"]; victim=f["victim"]; of=f["outer"]; tabu=f["tabu"]
        alive=f["alive"] if not is_last else list(G_max.nodes())

        # ── Graph ─────────────────────────────────────────────────────────────
        ax_g.clear(); ax_g.set_facecolor(C_BG)
        ax_g.set_aspect("equal"); ax_g.axis("off")
        ax_g.set_xlim(XLIM); ax_g.set_ylim(YLIM)

        # Ghost (already removed)
        ghost=[n for n in G_max.nodes() if n not in alive] if not is_last else []
        if ghost:
            gsg=G_max.subgraph(ghost); gps={n:pos[n] for n in ghost}
            nx.draw_networkx_edges(gsg,gps,ax=ax_g,edge_color=C_GHOST,width=0.7,alpha=0.25)
            nx.draw_networkx_nodes(gsg,gps,ax=ax_g,nodelist=ghost,
                                   node_color=C_GHOST,node_size=160,
                                   edgecolors="#BDBDBD",linewidths=0.4)
            nx.draw_networkx_labels(gsg,gps,ax=ax_g,font_size=5.5,font_color="#BDBDBD")

        # Active
        sg2=G_max.subgraph(alive); ps={n:pos[n] for n in alive}

        # Face fills
        for face in tris:
            if all(n in alive for n in face):
                in_of=all(n in of for n in face)
                fc="#FFF3E0" if in_of else "#F5F5F5"
                pts=np.array([pos[n] for n in face])
                ax_g.add_patch(Polygon(pts,closed=True,facecolor=fc,edgecolor="none",alpha=0.5))

        # Edges
        e_new=[e for e in sg2.edges() if tuple(sorted(e)) not in ori_edges]
        e_ori=[e for e in sg2.edges() if tuple(sorted(e)) in ori_edges]
        if e_new: nx.draw_networkx_edges(sg2,ps,ax=ax_g,edgelist=e_new,
                                          edge_color="#B2DFDB",width=0.8,alpha=0.4)
        if e_ori: nx.draw_networkx_edges(sg2,ps,ax=ax_g,edgelist=e_ori,
                                          edge_color=C_EDGE,width=1.4,alpha=0.7)

        # Nodes
        for n in alive:
            c_idx=col.get(n,-1); nc_=PAL.get(c_idx,C_GRAY)
            sz=280; bc="white"; blw=1.0
            if n==victim and not is_last: sz=480; blw=3.0
            elif n in of and not is_last: bc=C_OUT; blw=2.2; sz=310
            nx.draw_networkx_nodes(sg2,ps,ax=ax_g,nodelist=[n],
                                   node_color=nc_,node_size=sz,
                                   edgecolors=bc,linewidths=blw)

        fs=max(5.5,8.5-len(alive)*0.1)
        if ps: nx.draw_networkx_labels(sg2,ps,ax=ax_g,font_size=fs,
                                        font_color="white",font_weight="bold")

        # ── Legend ────────────────────────────────────────────────────────────
        ax_l.clear(); ax_l.set_facecolor(C_PANEL); ax_l.axis("off")
        ax_l.text(0.5,0.97,"Tabu Coloring",transform=ax_l.transAxes,
                  ha="center",va="top",fontsize=11,fontweight="bold",color="#1A237E")
        ax_l.text(0.5,0.905,graph_name,transform=ax_l.transAxes,
                  ha="center",va="top",fontsize=8.5,color="#546E7A",style="italic")
        y=0.84
        step_n=idx+1
        ax_l.text(0.5,y,"DONE ✓" if is_last else f"Step {step_n} / {len(steps)}",
                  transform=ax_l.transAxes,ha="center",va="top",fontsize=10,
                  fontweight="bold",color="#1B5E20" if is_last else "#E65100")
        y-=0.08
        if not is_last and victim is not None:
            ax_l.text(0.5,y,f"Node {victim}",transform=ax_l.transAxes,
                      ha="center",va="top",fontsize=9.5,fontweight="bold",color="#263238")
            y-=0.06
            ax_l.text(0.5,y,f"Outer face: {sorted(of)}",transform=ax_l.transAxes,
                      ha="center",va="top",fontsize=7.5,color="#546E7A")
            y-=0.055
            ax_l.text(0.5,y,f"Tabu: {tabu}",transform=ax_l.transAxes,
                      ha="center",va="top",fontsize=8,color="#C62828",fontweight="bold")
            y-=0.055
            ax_l.text(0.5,y,f"→ color {f['color']}  ({CNAMES.get(f['color'],'?')})",
                      transform=ax_l.transAxes,ha="center",va="top",fontsize=8.5,
                      fontweight="bold",color=PAL.get(f['color'],"#333"))
            y-=0.07
        ax_l.axhline(y+0.01,xmin=0.05,xmax=0.95,color="#B0BEC5",lw=0.8)
        y-=0.04
        ax_l.text(0.5,y,"Colors used:",transform=ax_l.transAxes,
                  ha="center",va="top",fontsize=8,color="#546E7A")
        y-=0.065
        used=sorted(set(col.values()))
        counts={c:sum(1 for x in col.values() if x==c) for c in used}
        for ci in used[:6]:
            r=FancyBboxPatch((0.08,y-0.038),0.84,0.062,boxstyle="round,pad=0.01",
                             facecolor=PAL.get(ci,"#777"),edgecolor="none",transform=ax_l.transAxes)
            ax_l.add_patch(r)
            ax_l.text(0.5,y,f"{CNAMES.get(ci,'?')}  ×{counts.get(ci,0)}",
                      transform=ax_l.transAxes,ha="center",va="center",
                      fontsize=7.5,color="white",fontweight="bold")
            y-=0.075
        ax_l.axhline(0.07,xmin=0.05,xmax=0.95,color="#B0BEC5",lw=0.8)
        ax_l.text(0.5,0.04,f"Colored: {len(col)} / {G_max.number_of_nodes()}",
                  transform=ax_l.transAxes,ha="center",va="center",
                  fontsize=8,color="#546E7A")

        # ── Info bar ──────────────────────────────────────────────────────────
        ax_i.clear(); ax_i.set_facecolor(C_PANEL); ax_i.axis("off")
        if is_last:
            nc_f=len(set(col.values()))
            v_f=all(col.get(u,-1)!=col.get(v,-1) for u,v in G_max.edges() if u in col and v in col)
            title=f"Done  |  {nc_f} colors  |  Valid: {v_f}  |  No backtracking"
            body="Pick outer-face node with fewest forbidden colors → assign min free color → remove → repeat"
        else:
            title=(f"Step {step_n}/{len(steps)}  |  Node {victim}  in outer face "
                   f"{sorted(of)}  |  Tabu = {tabu}  →  color {f['color']} "
                   f"({CNAMES.get(f['color'],'?')})")
            body="Tabu(v) = colors of already-colored direct neighbors  |  Selection: min tabu size, then min ID"
        ax_i.text(0.012,0.68,title,transform=ax_i.transAxes,
                  fontsize=9,va="center",color="#1A237E",fontweight="bold")
        ax_i.text(0.012,0.22,body,transform=ax_i.transAxes,
                  fontsize=8,va="center",color="#546E7A")

    anim=FuncAnimation(fig,draw,frames=len(frame_data),interval=int(1000/fps),repeat=True)
    if output_path is None: output_path=f"{graph_name}_tabu.gif"
    Path(output_path).parent.mkdir(parents=True,exist_ok=True)
    anim.save(output_path,writer=PillowWriter(fps=fps),dpi=110)
    plt.close(fig)
    print(f"  Saved → {output_path}")
    return output_path

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _print_list():
    print(f"\n{'Name':<22} {'Description'}")
    print("─"*65)
    for key,(_,desc) in GRAPHS.items():
        print(f"  {key:<20} {desc}")
    print()

def _menu():
    print("\n╔══════════════════════════════════════╗")
    print("║   Tabu Coloring Visualizer           ║")
    print("╚══════════════════════════════════════╝")
    _print_list()
    key=input("  Graph name (Enter = errera): ").strip() or "errera"
    out=input(f"  Output file [{key}_tabu.gif]: ").strip() or f"{key}_tabu.gif"
    fps=input("  FPS [0.9]: ").strip()
    fps=float(fps) if fps else 0.9
    render_gif(key, out, fps)

if __name__=="__main__":
    if "--list" in sys.argv: _print_list()
    elif len(sys.argv)>1:
        name=sys.argv[1]
        out=sys.argv[2] if len(sys.argv)>2 else f"{name}_tabu.gif"
        render_gif(name, out)
    else:
        _menu()
