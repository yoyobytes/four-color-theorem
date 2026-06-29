"""
onion_peeling.py — v8
═════════════════════
Peeling Topológico  +  Coloreo con Cadenas de Kempe  +  Backtracking 4-color

TRES ACTOS
─────────
Acto 1 — Peeling topológico vértice a vértice (menor grado en cara exterior).
Acto 2 — Coloreo greedy inverso CON intentos de swap de Kempe integrados.
Acto 3 — Si el greedy terminó con >4 colores: backtracking CSP con MRV
          y forward checking que garantiza exactamente 4 colores.
          Muestra qué nodos cambiaron de color respecto al greedy.

LAYOUT
──────
  Panel izquierdo (75 %): grafo con nodos y aristas.
  Panel derecho  (25 %): leyenda de colores y estado actual.
  Barra inferior:         título del paso e info del swap.

Requisitos:
    pip install shapely scipy matplotlib networkx pillow
"""

import json, math, sys
from collections import deque
from itertools import combinations
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Point

# ── Paleta ───────────────────────────────────────────────────────────────────
PAL   = {0:"#2196F3", 1:"#4CAF50", 2:"#FF9800", 3:"#E91E63"}
NAMES = {0:"Azul", 1:"Verde", 2:"Naranja", 3:"Rosa"}
C_NONE   = "#CFD8DC"
C_VICTIM = "#E53935"
C_OUTER  = "#FB8C00"
C_INNER  = "#1976D2"
C_EORI   = "#607D8B"
C_ENEW   = "#B2DFDB"
C_OK     = "#00C853"
C_TANG   = "#FFD600"
C_BLOCK  = "#FF1744"
C_BG     = "#FAFAFA"
C_PANEL  = "#ECEFF1"


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA
# ══════════════════════════════════════════════════════════════════════════════
def cargar_grafo(ruta):
    with open(ruta) as f: d = json.load(f)
    G = nx.Graph(); pos = {}
    for k, v in d["vertices"].items():
        n = int(k); G.add_node(n); pos[n] = tuple(v)
    for a, b in d["aristas"]: G.add_edge(int(a), int(b))
    return G, pos


# ══════════════════════════════════════════════════════════════════════════════
# 2. MAXIMALIZACIÓN RESTRINGIDA
# ══════════════════════════════════════════════════════════════════════════════
def _cruza(pos, u, v, G):
    seg = LineString([pos[u], pos[v]]); pu, pv = Point(pos[u]), Point(pos[v])
    for a, b in G.edges():
        if a in (u,v) or b in (u,v): continue
        s2 = LineString([pos[a], pos[b]])
        if seg.intersects(s2):
            ix = seg.intersection(s2)
            if ix.geom_type == "Point" and (ix.equals(pu) or ix.equals(pv)): continue
            return True
    return False

def maximalizar(G, pos):
    cands = sorted([(u,v) for u,v in combinations(G.nodes(),2) if not G.has_edge(u,v)],
                   key=lambda e: G.degree(e[0])+G.degree(e[1]))
    added = 0
    for u, v in cands:
        if not _cruza(pos, u, v, G): G.add_edge(u,v); added += 1
    nL = sorted(pos); co = np.array([pos[n] for n in nL])
    try: h=len(ConvexHull(co).vertices); exp=3*len(nL)-6-(h-3); ok=G.number_of_edges()==exp
    except: exp="?"; ok=False
    print(f"    +{added} aristas  total={G.number_of_edges()}  esperado={exp}  {'✓' if ok else '✗'}\n")
    return G


# ══════════════════════════════════════════════════════════════════════════════
# 3. EMBEDDING DE TUTTE
# ══════════════════════════════════════════════════════════════════════════════
def _outer_face(G):
    if G.number_of_nodes() <= 2: return list(G.nodes())
    ok, emb = nx.check_planarity(G)
    if not ok: return list(G.nodes())[:3]
    seen=set(); faces=[]
    for v in emb:
        for w in emb.neighbors(v):
            f=tuple(emb.traverse_face(v,w)); k=frozenset(f)
            if k not in seen: seen.add(k); faces.append(list(f))
    if not faces: return list(G.nodes())[:3]
    try:
        ecc=nx.eccentricity(G)
        outer=max(faces, key=lambda f: sum(ecc[v] for v in set(f)))
    except: outer=faces[0]
    return list(set(outer))

def tutte(G, scale=8.0):
    if G.number_of_nodes() <= 3:
        pos={}
        for i,n in enumerate(G.nodes()):
            a=2*math.pi*i/max(G.number_of_nodes(),1)
            pos[n]=(scale*math.cos(a), scale*math.sin(a))
        return pos
    outer=_outer_face(G); k=len(outer); nodes=sorted(G.nodes())
    pf={v:(scale*math.cos(2*math.pi*i/k+math.pi/2),
            scale*math.sin(2*math.pi*i/k+math.pi/2))
        for i,v in enumerate(outer)}
    inner=[v for v in nodes if v not in pf]
    idx={v:i for i,v in enumerate(inner)}; m=len(inner)
    if m==0: return pf
    A=np.zeros((m,m)); bx=np.zeros(m); by=np.zeros(m)
    for i,v in enumerate(inner):
        nb=list(G.neighbors(v)); A[i,i]=len(nb)
        for u in nb:
            if u in idx: A[i,idx[u]]=-1
            else: bx[i]+=pf[u][0]; by[i]+=pf[u][1]
    try:
        x=np.linalg.solve(A,bx); y=np.linalg.solve(A,by)
    except:
        pk=nx.kamada_kawai_layout(G,scale=scale)
        return {n:(round(float(a),4),round(float(b),4)) for n,(a,b) in pk.items()}
    pos=dict(pf)
    for i,v in enumerate(inner): pos[v]=(round(float(x[i]),4),round(float(y[i]),4))
    return pos


# ══════════════════════════════════════════════════════════════════════════════
# 4. CARA EXTERIOR (topológica)
# ══════════════════════════════════════════════════════════════════════════════
def outer_set(G):
    if G.number_of_nodes() <= 2: return set(G.nodes())
    return set(_outer_face(G))


# ══════════════════════════════════════════════════════════════════════════════
# 5. ACTO 1 — PEELING
# ══════════════════════════════════════════════════════════════════════════════
def build_peel_frames(G):
    frames=[]; sg=G.copy(); nv=sorted(G.nodes()); p=1
    while nv:
        outer=outer_set(sg); v=min(outer,key=lambda n:(sg.degree(n),n))
        frames.append({"outer":outer,"victim":v,"deg":sg.degree(v),
                        "vivos":list(nv),"step":p})
        print(f"  Paso {p:>3}: nodo {v:>3}  grado={sg.degree(v)}  cara_ext={len(outer)}")
        sg.remove_node(v); nv.remove(v); p+=1
    print(f"  Orden: {[f['victim'] for f in frames]}\n")
    return frames


def degeneracy_ordering(G):
    """
    Matula-Beck algorithm: repeatedly remove the globally minimum-degree vertex.
    Returns list of (node, degree_at_removal).
    O(V + E) with a bucket queue; here O(V²) for clarity.

    The degeneracy d(G) = max over all subgraphs of (minimum degree in subgraph).
    A d(G)-ordering guarantees greedy uses ≤ d(G)+1 colors.
    For planar graphs d(G) ≤ 5, so greedy on D-order uses ≤ 6 colors.
    """
    sg = G.copy(); order = []
    while sg.number_of_nodes() > 0:
        v = min(sg.nodes(), key=lambda n: (sg.degree(n), n))
        order.append((v, sg.degree(v)))
        sg.remove_node(v)
    return order


# ══════════════════════════════════════════════════════════════════════════════
# 6. ACTO 2 — COLOREO + KEMPE integrado
# ══════════════════════════════════════════════════════════════════════════════
def _kchain(G, color, start, c1, c2):
    chain=set(); q=deque([start])
    while q:
        v=q.popleft()
        if v in chain or color.get(v) not in (c1,c2): continue
        chain.add(v)
        for u in G.neighbors(v):
            if u in color and color[u] in (c1,c2) and u not in chain: q.append(u)
    return chain

def _valid_coloring(G, color):
    return all(color.get(u)!=color.get(v) for u,v in G.edges()
               if u in color and v in color)

def build_color_frames(G, peel_frames):
    """
    Builds color + kempe frames in one unified sequence.
    For each vertex (in reverse peel order):
      1. Insert vertex, assign greedy color → COLOR frame.
      2. If outer has 4 colors → KEMPE frame:
           analyze all (c1,c2) chains, classify, apply best swap if found.
    """
    orden=[f["victim"] for f in reversed(peel_frames)]
    color={}; sg=nx.Graph(); frames=[]

    for i, node in enumerate(orden):
        sg.add_node(node)
        for nb in G.neighbors(node):
            if nb in sg: sg.add_edge(node, nb)
        outer=outer_set(sg)
        forb={color[nb] for nb in G.neighbors(node) if nb in color}
        c=0
        while c in forb: c+=1
        color[node]=c
        oc={color[n] for n in outer if n in color}

        frames.append({
            "type":"color","step":i+1,"node":node,
            "outer":outer,"color":dict(color),"oc":oc,
            "forbidden":set(forb),"assigned":c,"vivos":list(sg.nodes()),
        })
        mark="✓" if len(oc)<=3 else "⚠"
        print(f"  [{i+1:>2}] node={node:>3}  →{c}  proh={sorted(forb)}"
              f"  outer_colors={sorted(oc)} {mark}")

        if len(oc) == 4:
            # Analyze Kempe chains
            best=None; cands=[]
            for c1, c2 in combinations(sorted(oc), 2):
                starts=[n for n in outer if color.get(n)==c1]
                for start in starts:
                    chain=_kchain(sg, color, start, c1, c2)
                    blocked=[n for n in outer if color.get(n)==c2 and n in chain]
                    nc={**color}
                    for v in chain: nc[v]=c2 if color[v]==c1 else c1
                    new_oc={nc[n] for n in outer if n in nc}
                    if blocked:
                        status="TANGLED"
                    elif _valid_coloring(sg, nc) and len(new_oc)<4:
                        status="SUCCESS"
                        if best is None:
                            best={"c1":c1,"c2":c2,"chain":sorted(chain),
                                  "new_color":nc,"new_oc":sorted(new_oc)}
                    else:
                        status="NOHELP"
                    cands.append({"c1":c1,"c2":c2,"start":start,
                                   "chain":sorted(chain),"blocked":blocked,"status":status})

            if best:
                color=best["new_color"]
                oc_after=sorted({color[n] for n in outer if n in color})
                print(f"       ↺ Kempe c{best['c1']}↔c{best['c2']}"
                      f"  chain={best['chain']}  outer→{oc_after}")
            else:
                n_t=sum(1 for c in cands if c['status']=='TANGLED')
                print(f"       ✗ TODAS ENREDADAS ({n_t} tangled) — contraejemplo Kempe")

            frames.append({
                "type":"kempe","step":i+1,"node":node,
                "outer":outer,"color":dict(color),
                "oc_before":sorted(oc),"oc_after":sorted({color[n] for n in outer if n in color}),
                "cands":cands,"best":best,"vivos":list(sg.nodes()),
            })

    valid=all(color[u]!=color[v] for u,v in G.edges())
    nc=len(set(color.values()))
    kf=[f for f in frames if f['type']=='kempe']
    print(f"\n  Colores usados: {nc}  válido: {'✓' if valid else '✗'}")
    print(f"  Frames Kempe: {len(kf)}"
          f"  ({sum(1 for f in kf if f['best'])} swaps,"
          f" {sum(1 for f in kf if not f['best'])} enredados)\n")
    return frames


# ══════════════════════════════════════════════════════════════════════════════
# 7. ACTO 3 — BACKTRACKING CSP (garantía de 4 colores)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 7. ACTO 3 — BACKTRACKING CSP con trazado completo
# ══════════════════════════════════════════════════════════════════════════════

def four_color_backtrack_traced(G):
    """
    MRV + forward-checking backtracking that records every decision.

    Events logged per call:
      ASSIGN   v c  — color c assigned to v
      PRUNE    u c  — c removed from domain of u (forward check from neighbor)
      FAIL     v    — no color worked for v, backtracking
      UNASSIGN v c  — assignment of c to v undone
      RESTORE  u c  — c restored to domain of u
      DONE          — solution found

    Returns (color_dict, frames) where frames is a list of animation
    snapshots, one per ASSIGN / UNASSIGN / FAIL / DONE event.
    Prune events are bundled into the snapshot of the ASSIGN that caused them.
    """
    nodes   = list(G.nodes())
    color   = {}
    domains = {v: set(range(4)) for v in nodes}
    # pending_prunes accumulates prunes since last snapshot
    pending_prunes = []

    def snapshot(kind, v, c):
        return {
            "kind"    : kind,
            "node"    : v,
            "color"   : c,
            "color_snap" : dict(color),
            "domains"    : {k: set(s) for k, s in domains.items()},
            "pruned_this_step": list(pending_prunes),
        }

    frames = []

    def mrv():
        u = [v for v in nodes if v not in color]
        if not u: return None
        return min(u, key=lambda v: (len(domains[v]), -G.degree(v)))

    def forward_check(v, c):
        pruned = []
        for u in G.neighbors(v):
            if u not in color and c in domains[u]:
                domains[u].discard(c)
                pruned.append(u)
                pending_prunes.append((u, c))
                if not domains[u]:
                    for u2 in pruned: domains[u2].add(c)
                    pending_prunes.clear()
                    return False
        return pruned

    def backtrack():
        v = mrv()
        if v is None:
            frames.append(snapshot("DONE", None, None))
            return True

        for c in sorted(domains[v]):
            if any(color.get(u) == c for u in G.neighbors(v)):
                continue
            pending_prunes.clear()
            pruned = forward_check(v, c)
            if pruned is False:
                continue
            color[v] = c
            frames.append(snapshot("ASSIGN", v, c))
            pending_prunes.clear()

            if backtrack(): return True

            del color[v]
            frames.append(snapshot("UNASSIGN", v, c))
            for u in pruned: domains[u].add(c)

        frames.append(snapshot("FAIL", v, None))
        return False

    success = backtrack()
    return (color if success else None), frames


def build_backtrack_frames(G, greedy_color, bt_color, bt_trace):
    """
    Builds animation frames for Act 3:
      • One "BT_INTRO" frame showing the greedy 5-color state
      • One frame per ASSIGN / UNASSIGN / FAIL / DONE event from the trace
    """
    extra_color   = max(greedy_color.values())
    nodes_extra   = {v for v, c in greedy_color.items() if c == extra_color}
    nodes_changed = {v for v in G.nodes() if greedy_color[v] != bt_color[v]}
    all_nodes     = list(G.nodes())

    print(f"  Nodos con color extra (#{extra_color}): {sorted(nodes_extra)}")
    print(f"  Nodos que cambian: {sorted(nodes_changed)} ({len(nodes_changed)})")
    print(f"  Frames de traza: {len(bt_trace)}\n")

    frames = [{
        "type"        : "bt_intro",
        "color"       : dict(greedy_color),
        "nodes_extra" : nodes_extra,
        "extra_color" : extra_color,
        "vivos"       : all_nodes,
    }]

    for event in bt_trace:
        frames.append({
            "type"             : "bt_trace",
            "kind"             : event["kind"],
            "node"             : event["node"],
            "assigned_color"   : event["color"],
            "color_snap"       : event["color_snap"],
            "domains"          : event["domains"],
            "pruned_this_step" : event["pruned_this_step"],
            "vivos"            : all_nodes,
            "nodes_changed"    : nodes_changed,
        })

    return frames


# ══════════════════════════════════════════════════════════════════════════════
# 8. ANIMACIÓN
# ══════════════════════════════════════════════════════════════════════════════


def _info_box(ax_info, line1, line2="", col2="#333333"):
    ax_info.clear(); ax_info.axis("off"); ax_info.set_facecolor(C_PANEL)
    ax_info.text(0.012, 0.72, line1, transform=ax_info.transAxes,
                 fontsize=9.5, fontweight="bold", va="center", color="#1A237E")
    if line2:
        ax_info.text(0.012, 0.28, line2, transform=ax_info.transAxes,
                     fontsize=8.5, va="center", color=col2)

def _legend_panel(ax_leg, color_counts=None, extra_items=None, act=1):
    ax_leg.clear(); ax_leg.axis("off"); ax_leg.set_facecolor(C_PANEL)
    y=0.97
    # Node role legend (always shown)
    ax_leg.text(0.5, y, "Nodos", transform=ax_leg.transAxes,
                ha="center", va="top", fontsize=9, fontweight="bold", color="#1A237E")
    y-=0.07
    if act == 1:
        role_items = [
            (C_VICTIM, "Víctima (a eliminar)"),
            (C_OUTER,  "Cara exterior"),
            (C_INNER,  "Interior"),
        ]
    else:
        role_items = [
            (C_OUTER, "Cara exterior activa"),
        ]
    for col, lbl in role_items:
        r=mpatches.FancyBboxPatch((0.06,y-0.035),0.88,0.062,
                                   boxstyle="round,pad=0.01",
                                   facecolor=col,edgecolor="none",
                                   transform=ax_leg.transAxes)
        ax_leg.add_patch(r)
        ax_leg.text(0.5, y, lbl, transform=ax_leg.transAxes,
                    ha="center",va="center",fontsize=8,color="white",fontweight="bold")
        y-=0.082
    y-=0.015
    # Color palette
    ax_leg.text(0.5, y, "Colores", transform=ax_leg.transAxes,
                ha="center", va="top", fontsize=9, fontweight="bold", color="#1A237E")
    y-=0.07
    for i in range(4):
        cnt = f" ×{color_counts.get(i,0)}" if color_counts else ""
        r=mpatches.FancyBboxPatch((0.06,y-0.035),0.88,0.062,
                                   boxstyle="round,pad=0.01",
                                   facecolor=PAL[i],edgecolor="none",
                                   transform=ax_leg.transAxes)
        ax_leg.add_patch(r)
        ax_leg.text(0.5, y, f"Color {i}  ({NAMES[i]}){cnt}",
                    transform=ax_leg.transAxes,ha="center",va="center",
                    fontsize=8,color="white",fontweight="bold")
        y-=0.082
    y-=0.015
    if extra_items:
        ax_leg.text(0.5, y, "Cadenas Kempe", transform=ax_leg.transAxes,
                    ha="center", va="top", fontsize=9, fontweight="bold", color="#1A237E")
        y-=0.07
        items=[
            (C_OK,    "white", "Cadena exitosa"),
            (C_TANG,  "#333",  "Cadena enredada"),
            (C_BLOCK, "white", "Nodo bloqueador"),
        ]
        for col, tc, lbl in items:
            r=mpatches.FancyBboxPatch((0.06,y-0.035),0.88,0.062,
                                       boxstyle="round,pad=0.01",
                                       facecolor=col,edgecolor="none",
                                       transform=ax_leg.transAxes)
            ax_leg.add_patch(r)
            ax_leg.text(0.5, y, lbl, transform=ax_leg.transAxes,
                        ha="center",va="center",fontsize=7.5,
                        color=tc,fontweight="bold")
            y-=0.078

def make_figure():
    fig=plt.figure(figsize=(11,8.5))
    fig.patch.set_facecolor(C_BG)
    ax_g   = fig.add_axes([0.01, 0.13, 0.70, 0.84])   # graph
    ax_leg = fig.add_axes([0.72, 0.13, 0.27, 0.84])   # legend
    ax_inf = fig.add_axes([0.01, 0.00, 0.98, 0.12])   # info bar
    ax_g.set_facecolor(C_BG); ax_g.set_aspect("equal"); ax_g.axis("off")
    return fig, ax_g, ax_leg, ax_inf

def generar_animacion(G_orig, G_max, peel_frames, color_frames, bt_frames=None,
                      ruta="descomposicion_grafo.gif", fps=1):

    pos       = tutte(G_max, scale=8.0)
    ori_edges = {tuple(sorted(e)) for e in G_orig.edges()}
    n1        = len(peel_frames)
    n2        = len(color_frames)
    n3        = len(bt_frames) if bt_frames else 0
    total     = n1 + n2 + n3

    fig, ax_g, ax_leg, ax_inf = make_figure()

    # ── pre-compute triangular faces of the full maximal graph ───────────────
    _ok, _emb = nx.check_planarity(G_max)
    _seen = set(); _all_faces = []
    for _v in _emb:
        for _w in _emb.neighbors(_v):
            _f = tuple(_emb.traverse_face(_v, _w))
            _k = frozenset(_f)
            if _k not in _seen:
                _seen.add(_k)
                _all_faces.append(list(set(_f)))
    _tri_faces = [f for f in _all_faces if len(f) == 3]  # triangular faces only

    # ── helpers ──────────────────────────────────────────────────────────────
    def draw_graph(vivos, node_styles, highlight_edges=None):
        """
        node_styles: dict {node: (color, size, edge_color, edge_lw)}
        Draws: face fills → edges → nodes → labels (back to front).
        """
        sg  = G_max.subgraph(vivos)
        ps  = {n: pos[n] for n in vivos}
        ax_g.clear(); ax_g.set_facecolor(C_BG); ax_g.set_aspect("equal"); ax_g.axis("off")

        # face fills (drawn first, behind everything)
        for face in _tri_faces:
            if all(n in vivos for n in face):
                pts = np.array([pos[n] for n in face])
                poly = plt.Polygon(pts, closed=True,
                                   facecolor="#E8EAF6", edgecolor="none",
                                   alpha=0.55, zorder=0)
                ax_g.add_patch(poly)

        # edges
        en=[e for e in sg.edges() if tuple(sorted(e)) not in ori_edges]
        eo=[e for e in sg.edges() if tuple(sorted(e)) in ori_edges]
        if en: nx.draw_networkx_edges(sg,ps,ax=ax_g,edgelist=en,
                                      edge_color=C_ENEW,width=0.9,alpha=0.45)
        if eo: nx.draw_networkx_edges(sg,ps,ax=ax_g,edgelist=eo,
                                      edge_color=C_EORI,width=1.6,alpha=0.75)
        if highlight_edges:
            for elist,col,lw in highlight_edges:
                valid_e=[(u,v) for u,v in elist if sg.has_edge(u,v)]
                if valid_e:
                    nx.draw_networkx_edges(sg,ps,ax=ax_g,edgelist=valid_e,
                                           edge_color=col,width=lw,alpha=0.9)

        # nodes (drawn on top of faces and edges)
        for n in vivos:
            nc,sz,bc,blw = node_styles.get(n,(C_NONE,280,"white",1.0))
            nx.draw_networkx_nodes(sg,ps,ax=ax_g,nodelist=[n],
                                   node_color=nc,node_size=sz,
                                   edgecolors=bc,linewidths=blw)

        # labels — adjust size based on node count
        fs = max(5.5, 8.5 - len(vivos)*0.12)
        nx.draw_networkx_labels(sg,ps,ax=ax_g,font_size=fs,
                                font_color="white",font_weight="bold")

    # ── ACTO 1 frames ────────────────────────────────────────────────────────
    def render_peel(idx):
        f      = peel_frames[idx]
        vivos  = f["vivos"]; outer = f["outer"]; v = f["victim"]

        styles = {}
        for n in vivos:
            if n == v:
                styles[n] = (C_VICTIM, 480, "white", 2.5)
            elif n in outer:
                styles[n] = (C_OUTER, 300, "white", 1.2)
            else:
                styles[n] = (C_INNER, 280, "white", 1.0)
        draw_graph(vivos, styles)

        _legend_panel(ax_leg, act=1)
        _info_box(ax_inf,
                  f"ACTO 1 — Peeling   paso {f['step']}/{n1}",
                  f"Eliminando nodo {v}   (grado mínimo = {f['deg']} en cara exterior)"
                  f"   |   cara exterior: {len(outer)} nodos")

    # ── ACTO 2 frames ────────────────────────────────────────────────────────
    def render_color(f):
        color = f["color"]; vivos = f["vivos"]
        outer = f["outer"]; node  = f["node"]
        oc    = f["oc"];    step  = f["step"]

        styles = {}
        for n in vivos:
            c = color.get(n, -1)
            col = PAL.get(c, C_NONE)
            sz=300; bc="white"; blw=1.2
            if n == node: sz=460; blw=2.5
            styles[n]=(col,sz,bc,blw)
        draw_graph(vivos, styles)

        cc = {i: sum(1 for x in color.values() if x==i) for i in range(4)}
        _legend_panel(ax_leg, color_counts=cc, act=2)

        ok  = len(oc) <= 3
        l2  = (f"Prohibidos: {sorted(f['forbidden'])}   →   asignado: color {f['assigned']}  ({NAMES.get(f['assigned'],'?')})"
               f"   |   outer colors: {sorted(oc)}")
        col2 = "#27AE60" if ok else "#E65100"
        mark = "✓  cara exterior ≤ 3 colores" if ok else "⚠  cara exterior usa 4 colores — intentando Kempe..."
        _info_box(ax_inf,
                  f"ACTO 2 — Coloreo   paso {step}/{n2 - sum(1 for x in color_frames if x['type']=='kempe')}   |   {mark}",
                  l2, col2)

    def render_kempe(f):
        color = f["color"]; vivos = f["vivos"]
        outer = f["outer"]; node  = f["node"]
        cands = f["cands"]; best  = f["best"]
        step  = f["step"]

        # Collect chain sets
        ok_chains   = [c for c in cands if c["status"]=="SUCCESS"]
        tang_chains = [c for c in cands if c["status"]=="TANGLED"]

        ok_nodes   = set()
        tang_nodes = set()
        block_nodes= set()
        ok_edges   = []
        tang_edges = []

        if best:
            ok_nodes=set(best["chain"])
            for n1_,n2_ in combinations(best["chain"],2):
                if G_max.has_edge(n1_,n2_): ok_edges.append((n1_,n2_))
        else:
            # Show biggest tangled chain
            if tang_chains:
                biggest=max(tang_chains,key=lambda c:len(c["chain"]))
                tang_nodes=set(biggest["chain"]); block_nodes=set(biggest["blocked"])
                for n1_,n2_ in combinations(biggest["chain"],2):
                    if G_max.has_edge(n1_,n2_): tang_edges.append((n1_,n2_))

        styles = {}
        for n in vivos:
            c   = color.get(n,-1)
            col = PAL.get(c,C_NONE)
            sz=300; bc="white"; blw=1.2
            if n in block_nodes:    bc=C_BLOCK;  blw=3.0; sz=340
            elif n in ok_nodes:     bc=C_OK;     blw=3.0; sz=340
            elif n in tang_nodes:   bc=C_TANG;   blw=2.5; sz=320
            if n==node:             sz=460;      blw=2.5
            styles[n]=(col,sz,bc,blw)

        hl=[]
        if ok_edges:   hl.append((ok_edges,   C_OK,   2.5))
        if tang_edges: hl.append((tang_edges, C_TANG, 2.0))
        draw_graph(vivos, styles, highlight_edges=hl if hl else None)

        cc={i:sum(1 for x in color.values() if x==i) for i in range(4)}
        _legend_panel(ax_leg, color_counts=cc, extra_items=True, act=2)

        n_t=sum(1 for c in cands if c["status"]=="TANGLED")
        n_s=sum(1 for c in cands if c["status"]=="SUCCESS")

        if best:
            title=(f"ACTO 2 — Kempe   paso {step}   |"
                   f"  ↺ swap c{best['c1']}↔c{best['c2']}   cadena: {best['chain']}")
            l2=(f"outer: {f['oc_before']}  →  {f['oc_after']}"
                f"   |   {n_s} cadenas exitosas,  {n_t} enredadas")
            col2="#27AE60"
        else:
            title=(f"ACTO 2 — Kempe   paso {step}   |"
                   f"  ✗  TODAS LAS CADENAS ENREDADAS — contraejemplo al algoritmo de Kempe")
            l2=(f"outer: {f['oc_before']}  — no existe swap que reduzca a ≤ 3 colores"
                f"   |   {n_t} cadenas enredadas,  {n_s} exitosas")
            col2="#C62828"
        _info_box(ax_inf, title, l2, col2)

    # ── ACTO 3 ───────────────────────────────────────────────────────────────
    def render_bt(f):
        all_nodes = f["vivos"]

        if f["type"] == "bt_intro":
            # Show the flawed greedy state before backtracking
            color    = f["color"]
            special  = f["nodes_extra"]
            styles   = {}
            for n in all_nodes:
                c = color.get(n, -1)
                col = PAL.get(c, C_NONE)
                # 5th-color nodes: larger with white border
                if n in special: styles[n] = (col, 460, "white", 3.5)
                else:            styles[n] = (col, 300, "white", 1.2)
            draw_graph(all_nodes, styles)
            cc = {i: sum(1 for x in color.values() if x==i) for i in range(5)}
            _legend_panel(ax_leg, color_counts={i:cc.get(i,0) for i in range(4)}, act=2)
            _info_box(ax_inf,
                      f"ACTO 3 — Backtracking CSP  |  "
                      f"Greedy usó {f['extra_color']+1} colores — activando backtracking",
                      f"Nodos con color extra (borde blanco): {sorted(special)}  |  "
                      f"MRV + forward checking garantizan 4 colores",
                      "#C62828")
            return

        # bt_trace frame
        kind     = f["kind"]
        node     = f["node"]
        col_snap = f["color_snap"]
        domains  = f["domains"]
        pruned   = f["pruned_this_step"]

        styles = {}
        for n in all_nodes:
            c   = col_snap.get(n, -1)
            col = PAL.get(c, C_NONE)
            sz  = 300; bc = "white"; blw = 1.2

            if n == node:
                if   kind == "ASSIGN":   bc = "#FFFFFF"; blw = 3.0; sz = 460
                elif kind == "UNASSIGN": col = "#E53935"; blw = 3.0; sz = 460
                elif kind == "FAIL":     col = "#B71C1C"; blw = 3.0; sz = 460
            elif n in [u for u, _ in pruned]:
                # pruned neighbor — dashed orange border indication
                bc = C_OUTER; blw = 2.2; sz = 320
            styles[n] = (col, sz, bc, blw)

        draw_graph(all_nodes, styles)

        # Domain annotations: show remaining domain size below each uncolored node
        sg2 = G_max.subgraph(all_nodes)
        ps2 = {n: pos[n] for n in all_nodes}
        for n in all_nodes:
            if n not in col_snap and domains:
                dom = domains.get(n, set())
                x, y = pos[n]
                ax_g.text(x, y - 0.55, str(len(dom)),
                          ha="center", va="center", fontsize=6,
                          color="#FF6F00", fontweight="bold")

        cc = {i: sum(1 for x in col_snap.values() if x==i) for i in range(4)}
        n_colored = len(col_snap)
        n_total   = len(all_nodes)
        _legend_panel(ax_leg, color_counts=cc, act=2)

        if kind == "ASSIGN":
            title = (f"ACTO 3 — Backtracking  |  "
                     f"ASSIGN  nodo {node} → color {col_snap.get(node,'?')}  "
                     f"({n_colored}/{n_total} coloreados)")
            l2 = (f"Forward check: {len(pruned)} dominios podados  |  "
                  f"MRV next: elige el nodo con menos opciones restantes")
            col2 = "#1B5E20"
        elif kind == "UNASSIGN":
            title = (f"ACTO 3 — Backtracking  |  "
                     f"UNASSIGN  nodo {node}  ← retrocediendo")
            l2 = "Ningún color funcionó para el siguiente nodo — deshaciendo asignación"
            col2 = "#E65100"
        elif kind == "FAIL":
            title = (f"ACTO 3 — Backtracking  |  "
                     f"FAIL  nodo {node}  — todos los colores bloqueados")
            l2 = "Los 4 colores están prohibidos por vecinos ya coloreados"
            col2 = "#C62828"
        elif kind == "DONE":
            nc = len(set(col_snap.values()))
            chg = f["nodes_changed"]
            title = (f"ACTO 3 — Backtracking  |  "
                     f"✓ SOLUCIÓN  {nc} colores — válido por el Teorema de los 4 Colores")
            l2 = (f"Nodos que cambiaron respecto al greedy: {sorted(chg)} "
                  f"({len(chg)} total)")
            col2 = "#1B5E20"
            # re-draw with changed nodes highlighted
            for n in all_nodes:
                c   = col_snap.get(n, -1)
                col = PAL.get(c, C_NONE)
                bc = "white"; blw = 1.2; sz = 300
                if n in chg: bc = "white"; blw = 3.5; sz = 420
                styles[n] = (col, sz, bc, blw)
            draw_graph(all_nodes, styles)
        else:
            title = f"ACTO 3 — Backtracking  |  {kind}  nodo {node}"
            l2 = ""; col2 = "#333"

        _info_box(ax_inf, title, l2, col2)

    # ── RENDER ───────────────────────────────────────────────────────────────
    def dibujar(idx):
        if idx < n1:
            render_peel(idx)
        elif idx < n1 + n2:
            f = color_frames[idx - n1]
            if f["type"] == "color": render_color(f)
            else:                    render_kempe(f)
        else:
            render_bt(bt_frames[idx - n1 - n2])

    anim = FuncAnimation(fig, dibujar, frames=total,
                         interval=int(1000/fps), repeat=True)
    anim.save(ruta, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    bt_note = f" + {n3} backtrack" if n3 else ""
    print(f"  ✓  '{ruta}'  ({total} frames: {n1} peel + {n2} coloreo{bt_note})")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ruta = Path("grafo.json")
    if not ruta.exists(): print(f"ERROR: '{ruta}' no encontrado."); sys.exit(1)

    sep="═"*60
    print(sep); print("  PEELING + COLOREO + KEMPE + BACKTRACKING  (v8)"); print(sep)

    print(f"\n[1] Cargando '{ruta}' …")
    G_orig, pos = cargar_grafo(str(ruta))
    print(f"    {G_orig.number_of_nodes()} vértices  {G_orig.number_of_edges()} aristas")
    if not nx.check_planarity(G_orig)[0]:
        print("  ✗ No planar."); sys.exit(1)
    print("    ✓ Planar\n")

    print("[2] Maximalización …")
    G_max = maximalizar(G_orig.copy(), pos)

    print("[3] Tutte layout …")
    print(f"    Cara exterior: {sorted(_outer_face(G_max))}\n")

    print("[4] D-ordering (Matula-Beck) …")
    d_order = degeneracy_ordering(G_max)
    d_nodes = [v for v, _ in d_order]
    d_degs  = [d for _, d in d_order]
    degeneracy = max(d_degs)
    print(f"    Orden D: {d_nodes}")
    print(f"    Degeneracy d(G) = {degeneracy}  →  greedy garantiza ≤ {degeneracy+1} colores\n")

    print("[5] ACTO 1 — Peeling topológico\n")
    pf = build_peel_frames(G_max)

    peel_nodes = [f["victim"] for f in pf]
    same = peel_nodes == d_nodes
    print(f"    Peeling topológico == D-ordering: {same}")
    if not same:
        diffs = [(i+1, peel_nodes[i], d_nodes[i])
                 for i in range(len(peel_nodes)) if peel_nodes[i] != d_nodes[i]]
        print(f"    Diferencias en pasos: {diffs}\n")
    else:
        print()

    print("[6] ACTO 2 — Coloreo + Kempe\n")
    cf = build_color_frames(G_max, pf)

    greedy_color = cf[-1]["color"]
    nc_greedy    = len(set(greedy_color.values()))

    bt_frames = None
    if nc_greedy > 4:
        print(f"[7] ACTO 3 — Backtracking CSP  (greedy usó {nc_greedy} colores)\n")
        bt_color, bt_trace = four_color_backtrack_traced(G_max)
        if bt_color:
            valid_bt = all(bt_color[u]!=bt_color[v] for u,v in G_max.edges())
            print(f"  Backtrack: {len(set(bt_color.values()))} colores  "
                  f"válido={'✓' if valid_bt else '✗'}  "
                  f"frames de traza: {len(bt_trace)}")
            bt_frames = build_backtrack_frames(G_max, greedy_color, bt_color, bt_trace)
        else:
            print("  ✗ Backtrack falló (no debería ocurrir para grafos planares)")
        step = "[8]"
    else:
        print(f"  Greedy ya usó {nc_greedy} colores — backtracking no necesario\n")
        step = "[7]"

    print(f"{step} Generando GIF …")
    generar_animacion(G_orig, G_max, pf, cf, bt_frames)

    final_color = bt_frames[-1]["color_snap"] if bt_frames else greedy_color
    nc_final    = len(set(final_color.values()))
    val_final   = all(final_color[u]!=final_color[v] for u,v in G_max.edges())
    print(f"\n  Colores finales: {nc_final}  |  Válido: {'✓' if val_final else '✗'}")
    print(f"\n¡Listo!"); print(sep)

if __name__ == "__main__": main()
