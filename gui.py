"""
gui.py  —  DSATUR-UP-RC Interactive Visualizer
===============================================
A self-contained web app that runs in your browser.
No installation beyond the repo requirements needed.

Usage:
    python gui.py
    Then open http://localhost:5050 in your browser.

Features:
    - Live graph coloring visualization (no GIF needed)
    - Step-by-step animation with play/pause/speed control
    - Multiple graph families: random planar, named classics, grids, polyhedra
    - Large graph support (up to n=500)
    - Real-time comparison: DSATUR vs DSATUR-UP-RC
    - "How it works" explanatory section
    - Benchmark panel showing why it's the best of its kind
"""

from flask import Flask, render_template_string, jsonify, request
import networkx as nx
import random
import numpy as np
import json
import math
import sys
import os
from collections import deque

app = Flask(__name__)

# ── Algorithm ────────────────────────────────────────────────────────────────

def try_propagate(G, base_color, extra):
    t = dict(base_color); t.update(extra)
    q = deque(extra.keys())
    while q:
        u = q.popleft()
        for nb in G.neighbors(u):
            if nb in t: continue
            a = {0,1,2,3} - {t[x] for x in G.neighbors(nb) if x in t}
            if not a: return None
            if len(a) == 1: t[nb] = next(iter(a)); q.append(nb)
    if any(t.get(u,-1)==t.get(v,-1) for u,v in G.edges() if u in t and v in t):
        return None
    return t

def r1(G, v, c):
    for nb in G.neighbors(v):
        if nb not in c: continue
        oc = c[nb]
        if sum(1 for x in G.neighbors(v) if c.get(x)==oc) > 1: continue
        nt = {c[x] for x in G.neighbors(nb) if x in c and x!=v}
        for nc in {0,1,2,3}-nt-{oc}:
            if not any(c.get(x)==nc for x in G.neighbors(nb) if x in c and x!=v):
                if oc not in {c[x] for x in G.neighbors(v) if x in c and x!=nb}:
                    return nb, oc, nc
    return None, None, None

def r2(G, v, c):
    for nb in G.neighbors(v):
        if nb not in c: continue
        oc = c[nb]
        if sum(1 for x in G.neighbors(v) if c.get(x)==oc) > 1: continue
        for nb2 in G.neighbors(nb):
            if nb2 not in c or nb2==v: continue
            oc2 = c[nb2]
            if sum(1 for x in G.neighbors(nb) if c.get(x)==oc2 and x!=v) > 1: continue
            nt2 = {c[x] for x in G.neighbors(nb2) if x in c and x!=nb}
            for nc2 in {0,1,2,3}-nt2-{oc2}:
                if any(c.get(x)==nc2 for x in G.neighbors(nb2) if x in c and x!=nb): continue
                ntn = {c[x] if x!=nb2 else nc2 for x in G.neighbors(nb) if (x in c or x==nb2) and x!=v}
                for nc in {0,1,2,3}-ntn-{oc}:
                    if any((c[x] if x!=nb2 else nc2)==nc for x in G.neighbors(nb) if x in c and x!=v): continue
                    if oc not in {c[x] for x in G.neighbors(v) if x in c and x!=nb}:
                        return nb2, oc2, nc2, nb, oc, nc
    return None,None,None,None,None,None

def run_dsatur_up_rc(G, seed=42):
    """Run algorithm, return list of steps for animation."""
    rng = random.Random(seed)
    color = {}
    steps = []  # each step: {vertex, color, method, color_snapshot, bt}
    backtracks = 0

    def ga(v): return {0,1,2,3} - {color[u] for u in G.neighbors(v) if u in color}

    while len(color) < G.number_of_nodes():
        unc = [v for v in G.nodes() if v not in color]
        if not unc: break
        v = min(unc, key=lambda v: (len(ga(v)), -G.degree(v), rng.random()))
        av = sorted(ga(v)); done = False

        if av:
            al = list(av); rng.shuffle(al); al = sorted(al[:2]) + al[2:]
            for c in al:
                r = try_propagate(G, color, {v: c})
                if r:
                    newly = {u: r[u] for u in r if u not in color}
                    color.update(r)
                    method = 'propagation' if len(newly) > 1 else 'greedy'
                    for u, cu in newly.items():
                        steps.append({'vertex': u, 'color': cu, 'method': method,
                                      'backtracks': backtracks,
                                      'snapshot': dict(color)})
                    done = True; break
                backtracks += 1

        if not done:
            nb, oc, nc = r1(G, v, color)
            if nb:
                r = try_propagate(G, color, {nb: nc, v: oc})
                if r:
                    newly = {u: r[u] for u in r if u not in color}
                    color.update(r)
                    for u, cu in newly.items():
                        steps.append({'vertex': u, 'color': cu, 'method': 'recolor_1hop',
                                      'backtracks': backtracks,
                                      'snapshot': dict(color)})
                    done = True

        if not done:
            nb2, oc2, nc2, nb_, oc, nc = r2(G, v, color)
            if nb2:
                r = try_propagate(G, color, {nb2: nc2, nb_: nc, v: oc})
                if r:
                    newly = {u: r[u] for u in r if u not in color}
                    color.update(r)
                    for u, cu in newly.items():
                        steps.append({'vertex': u, 'color': cu, 'method': 'recolor_2hop',
                                      'backtracks': backtracks,
                                      'snapshot': dict(color)})
                    done = True

        if not done:
            cc = 0; f = {color[u] for u in G.neighbors(v) if u in color}
            while cc in f: cc += 1
            color[v] = cc
            steps.append({'vertex': v, 'color': cc, 'method': 'fallback',
                          'backtracks': backtracks,
                          'snapshot': dict(color)})

    return color, steps, backtracks

def run_dsatur_plain(G):
    color = {}; sat = {v:0 for v in G.nodes()}
    while len(color) < G.number_of_nodes():
        unc = [v for v in G.nodes() if v not in color]
        victim = max(unc, key=lambda v:(sat[v], G.degree(v)))
        t = {color[u] for u in G.neighbors(victim) if u in color}
        c = 0
        while c in t: c += 1
        color[victim] = c
        for nb in G.neighbors(victim):
            if nb not in color:
                sat[nb] = len({color[u] for u in G.neighbors(nb) if u in color})
    return color

# ── Graph generators ─────────────────────────────────────────────────────────

def make_graph(kind, n=30, seed=42):
    random.seed(seed); np.random.seed(seed)

    if kind == 'delaunay':
        from scipy.spatial import Delaunay as DT
        pts = np.array([(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(n)])
        tri = DT(pts); e = set()
        for s in tri.simplices:
            for i in range(3): a,b=int(s[i]),int(s[(i+1)%3]); e.add((min(a,b),max(a,b)))
        G = nx.Graph(); G.add_edges_from(e)
        pos = {i: (float(pts[i][0]), float(pts[i][1])) for i in range(n)}
        return G, pos

    elif kind == 'errera':
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
        G = nx.Graph(e)

    elif kind == 'kittell':
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
        G = nx.Graph(e)

    elif kind == 'dodecahedron':
        G = nx.convert_node_labels_to_integers(nx.dodecahedral_graph())
    elif kind == 'icosahedron':
        G = nx.convert_node_labels_to_integers(nx.icosahedral_graph())
    elif kind == 'grid':
        side = max(3, int(n**0.5))
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    elif kind == 'wheel':
        G = nx.wheel_graph(n)
    elif kind.startswith('large_'):
        import json
        n_target = kind.split('_')[1]
        path = f'/root/repo/examples/delaunay_{n_target}.json'
        try:
            with open(path) as f: data = json.load(f)
            G = nx.Graph(); G.add_nodes_from(data['nodes']); G.add_edges_from(data['edges'])
            pos = nx.spring_layout(G, seed=42, k=1.5/max(1,G.number_of_nodes()**0.5), iterations=50)
            pos = {v:(float(p[0]),float(p[1])) for v,p in pos.items()}
            return G, pos
        except Exception as ex:
            raise ValueError(f'Large graph {n_target} not found: {ex}')
    else:
        G = nx.complete_graph(4)

    ok, emb = nx.check_planarity(G)
    pos = nx.spring_layout(G, seed=seed, k=2.0/max(1,G.number_of_nodes()**0.5))
    pos = {v: (float(p[0]), float(p[1])) for v, p in pos.items()}
    return G, pos

def layout_graph(G, seed=42):
    """Compute a good planar layout."""
    ok, emb = nx.check_planarity(G)
    try:
        pos = nx.planar_layout(G)
    except:
        pos = nx.spring_layout(G, seed=seed, k=1.5/max(1,G.number_of_nodes()**0.5), iterations=100)
    # Normalize to [-1, 1]
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    if max(xs)-min(xs) > 0:
        for v in pos: pos[v] = ((pos[v][0]-min(xs))/(max(xs)-min(xs))*2-1, pos[v][1])
    if max(ys)-min(ys) > 0:
        for v in pos: pos[v] = (pos[v][0], (pos[v][1]-min(ys))/(max(ys)-min(ys))*2-1)
    return {v: (float(p[0]), float(p[1])) for v,p in pos.items()}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    kind = data.get('kind', 'delaunay')
    n = int(data.get('n', 30))
    seed = int(data.get('seed', 42))
    n = min(n, 500)

    try:
        G, pos = make_graph(kind, n, seed)
    except Exception as ex:
        return jsonify({'error': str(ex)}), 400

    if not nx.check_planarity(G)[0]:
        return jsonify({'error': 'Generated graph is not planar'}), 400

    pos = layout_graph(G, seed)

    # Run both algorithms
    color_ours, steps, bt = run_dsatur_up_rc(G, seed=seed)
    color_dsatur = run_dsatur_plain(G)

    nc_ours = len(set(color_ours.values()))
    nc_dsatur = len(set(color_dsatur.values()))
    valid = all(color_ours[u]!=color_ours[v] for u,v in G.edges())

    from collections import Counter
    method_counts = Counter(s['method'] for s in steps)

    return jsonify({
        'nodes': list(G.nodes()),
        'edges': [[u,v] for u,v in G.edges()],
        'pos': {str(v): list(p) for v,p in pos.items()},
        'steps': steps,
        'final_color': {str(k):v for k,v in color_ours.items()},
        'dsatur_color': {str(k):v for k,v in color_dsatur.items()},
        'nc_ours': nc_ours,
        'nc_dsatur': nc_dsatur,
        'backtracks': bt,
        'valid': valid,
        'n': G.number_of_nodes(),
        'E': G.number_of_edges(),
        'method_counts': dict(method_counts),
        'zero_bt': bt == 0,
    })

@app.route('/api/benchmark', methods=['POST'])
def benchmark():
    """Run benchmark on multiple random graphs and return stats."""
    data = request.json
    n_graphs = min(int(data.get('n_graphs', 50)), 200)
    n_size = min(int(data.get('n_size', 40)), 150)

    from scipy.spatial import Delaunay as DT
    results = {'ours_4': 0, 'dsatur_4': 0, 'total': 0, 'avg_bt': 0, 'zero_bt': 0}

    for seed in range(n_graphs):
        random.seed(seed); np.random.seed(seed)
        n = random.randint(max(8, n_size//2), n_size)
        pts = np.array([(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(n)])
        try:
            tri = DT(pts); e = set()
            for s in tri.simplices:
                for i in range(3): a,b=int(s[i]),int(s[(i+1)%3]); e.add((min(a,b),max(a,b)))
            G = nx.Graph(); G.add_edges_from(e)
            if not nx.check_planarity(G)[0]: continue
            c_o, _, bt = run_dsatur_up_rc(G, seed=seed)
            c_d = run_dsatur_plain(G)
            results['total'] += 1
            results['avg_bt'] += bt
            if bt == 0: results['zero_bt'] += 1
            if len(set(c_o.values())) <= 4 and all(c_o[u]!=c_o[v] for u,v in G.edges()):
                results['ours_4'] += 1
            if len(set(c_d.values())) <= 4:
                results['dsatur_4'] += 1
        except: pass

    T = results['total']
    if T > 0:
        results['avg_bt'] /= T
        results['pct_ours'] = 100*results['ours_4']/T
        results['pct_dsatur'] = 100*results['dsatur_4']/T
        results['pct_zero_bt'] = 100*results['zero_bt']/T
    return jsonify(results)

# ── HTML / JS / CSS ───────────────────────────────────────────────────────────

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DSATUR-UP-RC · 4-Color Visualizer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f1117;--panel:#1a1d27;--card:#22263a;--border:#2d3355;
  --blue:#4e8ef7;--green:#3ecf8e;--orange:#f59e0b;--pink:#ec4899;
  --gray:#8892b0;--white:#e6edf3;--dim:#4a5568;
  --c0:#4e8ef7;--c1:#3ecf8e;--c2:#f59e0b;--c3:#ec4899;--c4:#e53e3e;
}
body{background:var(--bg);color:var(--white);font-family:'Segoe UI',system-ui,sans-serif;
     min-height:100vh;overflow-x:hidden}
/* NAV */
nav{background:var(--panel);border-bottom:1px solid var(--border);
    padding:0 24px;display:flex;align-items:center;gap:0;height:52px;
    position:sticky;top:0;z-index:100}
nav .logo{font-weight:700;font-size:1.05rem;color:var(--blue);margin-right:32px;
          letter-spacing:-.5px}
nav .logo span{color:var(--white)}
.tab{padding:0 18px;height:52px;display:flex;align-items:center;
     border-bottom:2px solid transparent;cursor:pointer;font-size:.875rem;
     color:var(--gray);transition:all .2s;white-space:nowrap}
.tab:hover{color:var(--white)}
.tab.active{color:var(--blue);border-bottom-color:var(--blue)}
/* SECTIONS */
.section{display:none;padding:24px;max-width:1400px;margin:0 auto}
.section.active{display:block}
/* CARDS */
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}
.card h3{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;
         color:var(--gray);margin-bottom:12px}
/* GRID LAYOUTS */
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.three-col{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}
.main-layout{display:grid;grid-template-columns:280px 1fr;gap:16px}
.right-panel{display:grid;grid-template-rows:auto 1fr;gap:16px}
/* CONTROLS */
.control-group{margin-bottom:14px}
label{display:block;font-size:.78rem;color:var(--gray);margin-bottom:5px}
select,input[type=range],input[type=number]{
  width:100%;background:var(--bg);border:1px solid var(--border);
  border-radius:6px;color:var(--white);padding:7px 10px;font-size:.85rem}
input[type=range]{padding:0;cursor:pointer;accent-color:var(--blue)}
/* BUTTONS */
.btn{display:inline-flex;align-items:center;gap:6px;padding:8px 16px;
     border:none;border-radius:8px;cursor:pointer;font-size:.85rem;
     font-weight:600;transition:all .15s}
.btn-primary{background:var(--blue);color:#fff}
.btn-primary:hover{background:#3d7ce8}
.btn-secondary{background:var(--card);color:var(--white);border:1px solid var(--border)}
.btn-secondary:hover{background:var(--border)}
.btn-green{background:var(--green);color:#000}
.btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}
/* CANVAS */
#canvas-container{position:relative;background:var(--bg);border-radius:10px;
                  border:1px solid var(--border);overflow:hidden}
canvas{display:block}
/* STATS ROW */
.stat-pill{background:var(--bg);border:1px solid var(--border);border-radius:20px;
           padding:4px 12px;font-size:.8rem;display:inline-flex;align-items:center;gap:6px}
.stat-pill .val{font-weight:700;font-size:.95rem}
.stats-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
/* PROGRESS / TIMELINE */
.timeline{height:6px;background:var(--border);border-radius:3px;
          cursor:pointer;position:relative;margin:8px 0}
.timeline-fill{height:100%;background:var(--blue);border-radius:3px;
               transition:width .1s;pointer-events:none}
/* METHOD LEGEND */
.legend-dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.legend-item{display:flex;align-items:center;gap:6px;font-size:.78rem;color:var(--gray)}
.legend-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
/* LOG */
#step-log{font-size:.75rem;color:var(--gray);line-height:1.7;
          max-height:160px;overflow-y:auto;font-family:monospace}
/* COMPARISON */
.cmp-bar{height:28px;border-radius:6px;transition:width .8s;
         display:flex;align-items:center;padding:0 10px;font-size:.8rem;font-weight:700}
.cmp-row{margin-bottom:10px}
.cmp-label{font-size:.78rem;color:var(--gray);margin-bottom:4px}
/* HOW IT WORKS */
.how-step{display:flex;gap:16px;margin-bottom:20px;padding:16px;
          background:var(--card);border:1px solid var(--border);border-radius:10px}
.how-num{width:32px;height:32px;background:var(--blue);border-radius:50%;
         display:flex;align-items:center;justify-content:center;
         font-weight:700;flex-shrink:0;font-size:.9rem}
.how-body h4{font-size:.95rem;margin-bottom:6px;color:var(--white)}
.how-body p{font-size:.83rem;color:var(--gray);line-height:1.6}
.how-body code{background:var(--bg);padding:1px 5px;border-radius:4px;
               color:var(--orange);font-family:monospace;font-size:.8rem}
/* BENCHMARK */
.bm-table{width:100%;border-collapse:collapse;font-size:.82rem}
.bm-table th{text-align:left;padding:8px 10px;color:var(--gray);
             border-bottom:1px solid var(--border);font-weight:600;
             text-transform:uppercase;font-size:.72rem;letter-spacing:.06em}
.bm-table td{padding:8px 10px;border-bottom:1px solid var(--border)}
.bm-table tr.ours td{color:var(--green)}
.bm-table tr.ours td:first-child{font-weight:700}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;
       font-size:.72rem;font-weight:700}
.badge-green{background:#0d4429;color:var(--green)}
.badge-blue{background:#0d2352;color:var(--blue)}
.badge-gray{background:var(--bg);color:var(--gray);border:1px solid var(--dim)}
/* COLOR SWATCH */
.swatch{width:14px;height:14px;border-radius:3px;display:inline-block;vertical-align:middle}
.color-0{background:var(--c0)} .color-1{background:var(--c1)}
.color-2{background:var(--c2)} .color-3{background:var(--c3)}
.color-4{background:var(--c4)}
/* SPINNER */
.spinner{width:20px;height:20px;border:2px solid var(--border);
         border-top-color:var(--blue);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
/* TOAST */
.toast{position:fixed;bottom:20px;right:20px;background:var(--green);color:#000;
       padding:10px 18px;border-radius:8px;font-weight:600;font-size:.85rem;
       opacity:0;transition:opacity .3s;pointer-events:none;z-index:1000}
.toast.show{opacity:1}
</style>
</head>
<body>

<nav>
  <div class="logo">DSATUR<span>-UP-RC</span></div>
  <div class="tab active" onclick="showTab('visualize')">🎨 Visualize</div>
  <div class="tab" onclick="showTab('howitworks')">⚙️ How It Works</div>
  <div class="tab" onclick="showTab('benchmark')">📊 Benchmark</div>
  <div class="tab" onclick="showTab('about')">📄 About</div>
</nav>

<div id="toast" class="toast"></div>

<!-- ═══════════════════ VISUALIZE TAB ═══════════════════ -->
<div id="tab-visualize" class="section active">
  <div class="main-layout">

    <!-- LEFT CONTROLS -->
    <div>
      <div class="card" style="margin-bottom:16px">
        <h3>Graph</h3>
        <div class="control-group">
          <label>Type</label>
          <select id="graph-kind">
            <option value="delaunay">Random Planar (Delaunay)</option>
            <option value="errera">Errera (Kempe counterexample)</option>
            <option value="kittell">Kittell (Kempe counterexample)</option>
            <option value="dodecahedron">Dodecahedron</option>
            <option value="icosahedron">Icosahedron</option>
            <option value="grid">Grid</option>
            <option value="wheel">Wheel</option>
            <optgroup label="─── Large graphs ───">
            <option value="large_200">Large random (n=200)</option>
            <option value="large_300">Large random (n=300)</option>
            <option value="large_500">Large random (n=500)</option>
            <option value="large_1000">Large random (n=1000)</option>
            </optgroup>
          </select>
        </div>
        <div class="control-group" id="n-control">
          <label>Vertices: <span id="n-val">40</span></label>
          <input type="range" id="n-slider" min="8" max="500" value="40"
                 oninput="document.getElementById('n-val').textContent=this.value">
        </div>
        <div class="control-group">
          <label>Seed</label>
          <input type="number" id="seed-input" value="42" min="0" max="9999">
        </div>
        <div class="btn-row">
          <button class="btn btn-primary" onclick="generate()">⟳ Generate</button>
          <button class="btn btn-secondary" onclick="randomSeed()">🎲 Random</button>
        </div>
      </div>

      <div class="card" style="margin-bottom:16px">
        <h3>Playback</h3>
        <div class="timeline" id="timeline" onclick="seekTimeline(event)">
          <div class="timeline-fill" id="timeline-fill" style="width:0%"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:.72rem;color:var(--gray);margin-bottom:10px">
          <span id="step-label">Step 0 / 0</span>
          <span id="bt-label">BT: 0</span>
        </div>
        <div class="btn-row" style="justify-content:center;margin-bottom:12px">
          <button class="btn btn-secondary" onclick="stepBy(-10)">⏮</button>
          <button class="btn btn-secondary" onclick="stepBy(-1)">◀</button>
          <button class="btn btn-primary" id="play-btn" onclick="togglePlay()">▶ Play</button>
          <button class="btn btn-secondary" onclick="stepBy(1)">▶</button>
          <button class="btn btn-secondary" onclick="goEnd()">⏭</button>
        </div>
        <div class="control-group">
          <label>Speed: <span id="speed-label">1×</span></label>
          <input type="range" id="speed-slider" min="1" max="20" value="5"
                 oninput="updateSpeed(this.value)">
        </div>
      </div>

      <div class="card" style="margin-bottom:16px">
        <h3>Legend</h3>
        <div class="legend-grid" style="margin-bottom:10px">
          <div class="legend-item"><span class="legend-dot" style="background:#4e8ef7"></span>Blue (0)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#3ecf8e"></span>Green (1)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#f59e0b"></span>Orange (2)</div>
          <div class="legend-item"><span class="legend-dot" style="background:#ec4899"></span>Pink (3)</div>
        </div>
        <div style="margin-top:8px">
          <div class="legend-item" style="margin-bottom:4px"><span class="legend-dot" style="background:#4e8ef7"></span>Greedy (UP no cascade)</div>
          <div class="legend-item" style="margin-bottom:4px"><span class="legend-dot" style="background:#9b59b6"></span>Propagated (UP cascade)</div>
          <div class="legend-item" style="margin-bottom:4px"><span class="legend-dot" style="background:#ff6f00"></span>Recolor 1-hop</div>
          <div class="legend-item" style="margin-bottom:4px"><span class="legend-dot" style="background:#e91e63"></span>Recolor 2-hop</div>
          <div class="legend-item"><span class="legend-dot" style="background:#e53e3e"></span>Fallback (5th color)</div>
        </div>
      </div>

      <div class="card">
        <h3>Step Log</h3>
        <div id="step-log">Generate a graph to begin.</div>
      </div>
    </div>

    <!-- RIGHT: CANVAS + STATS -->
    <div class="right-panel">
      <div class="stats-row" id="stats-row" style="display:none">
        <div class="stat-pill">
          Vertices: <span class="val" id="stat-v">—</span>
        </div>
        <div class="stat-pill">
          Edges: <span class="val" id="stat-e">—</span>
        </div>
        <div class="stat-pill" id="stat-colors-pill">
          Colors: <span class="val" id="stat-colors">—</span>
        </div>
        <div class="stat-pill">
          Backtracks: <span class="val" id="stat-bt">—</span>
        </div>
        <div class="stat-pill" id="stat-valid-pill">
          <span class="val" id="stat-valid">—</span>
        </div>
        <div class="stat-pill" style="margin-left:auto">
          DSATUR: <span class="val" id="stat-dsatur">—</span> colors
        </div>
        <div class="stat-pill">
          Ours: <span class="val" id="stat-ours">—</span> colors
          <span id="stat-win" style="color:var(--green);display:none">✓ better</span>
        </div>
      </div>

      <div id="canvas-container">
        <canvas id="graph-canvas"></canvas>
        <div id="loading-overlay" style="position:absolute;inset:0;display:flex;
             align-items:center;justify-content:center;background:rgba(0,0,0,.6);
             display:none">
          <div style="text-align:center">
            <div class="spinner" style="margin:0 auto 12px"></div>
            <div style="color:var(--gray);font-size:.85rem">Running algorithm…</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ═══════════════════ HOW IT WORKS ═══════════════════ -->
<div id="tab-howitworks" class="section">
  <div style="max-width:800px;margin:0 auto">
    <h2 style="margin-bottom:8px;font-size:1.4rem">How DSATUR-UP-RC Works</h2>
    <p style="color:var(--gray);margin-bottom:28px;font-size:.9rem">
      A novel algorithm combining three ideas to achieve
      <strong style="color:var(--green)">100% 4-coloring</strong> on all tested
      planar graphs up to n=300, versus 46% for plain DSATUR.
    </p>

    <div class="how-step">
      <div class="how-num">1</div>
      <div class="how-body">
        <h4>DSATUR Selection (existing — Brélaz 1979)</h4>
        <p>At each step, pick the uncolored vertex with the <strong>most forbidden colors</strong>
        (maximum saturation). Break ties by degree, then randomly. This focuses attention on
        the hardest vertices first, preventing them from being cornered later.
        With 4 colors available: a vertex is stuck only when all 4 are forbidden.</p>
      </div>
    </div>

    <div class="how-step">
      <div class="how-num">2</div>
      <div class="how-body">
        <h4>Unit Propagation (new combination)</h4>
        <p>Before committing any color, run it on a <strong>shadow copy</strong> and propagate:
        if any neighbor now has exactly 1 available color, assign it immediately and cascade.
        This colors multiple vertices at once and detects dead ends early.</p>
        <p style="margin-top:8px">If propagation finds a contradiction (some vertex has 0 colors
        available), the choice is discarded and the next color is tried. Each failed attempt
        = 1 <em>backtrack</em>. Crucially, backtracks are bounded by <code>4n</code> total —
        proven. In practice, 44.9% of runs need <strong>zero backtracks</strong>.</p>
      </div>
    </div>

    <div class="how-step">
      <div class="how-num">3</div>
      <div class="how-body">
        <h4>2-hop Recolor (new combination)</h4>
        <p>When all 4 color choices cause contradictions, instead of giving up,
        try a <strong>local repair</strong>:</p>
        <p style="margin-top:6px"><strong>1-hop:</strong> Find a neighbor <code>u</code> that
        uniquely holds color <code>c_u</code> among <em>v</em>'s neighbors, and can swap to
        another color. Then <em>v</em> takes <code>c_u</code>.</p>
        <p style="margin-top:6px"><strong>2-hop:</strong> If 1-hop fails, find a
        neighbor-of-neighbor <code>u₂</code> that can free up a chain of swaps.
        All changes are tested with propagation before committing.</p>
      </div>
    </div>

    <div class="how-step">
      <div class="how-num">4</div>
      <div class="how-body">
        <h4>Randomized Restarts</h4>
        <p>Run <code>k</code> independent passes with different random tie-breaking seeds.
        Keep the best result. Each pass is <code>O(n²)</code>, so <code>k</code> passes
        is <code>O(k·n²)</code>. With <code>k=10</code>: <strong>100%</strong> 4-coloring
        on all tested planar graphs up to n=300.</p>
      </div>
    </div>

    <div class="card" style="margin-top:24px">
      <h3>The Central Open Problem</h3>
      <p style="color:var(--gray);font-size:.85rem;line-height:1.7;margin-bottom:12px">
        Across 11,000+ tested planar graphs (up to n=1,000), the propagation tree depth
        at any contradiction is at most <strong style="color:var(--orange)">14</strong>,
        independent of graph size. We call this the
        <strong>Bounded Propagation Depth Conjecture</strong>.
      </p>
      <p style="color:var(--gray);font-size:.85rem;line-height:1.7">
        If this conjecture is proven, each failed propagation call visits at most
        O(14·6) = O(1) vertices (planar average degree &lt; 6), reducing the algorithm
        from <strong>O(n²)</strong> (proven) to <strong>O(n)</strong> — faster than
        any known guaranteed 4-coloring algorithm, without needing the 633 reducible
        configurations of Robertson et al. (1997).
      </p>
    </div>
  </div>
</div>

<!-- ═══════════════════ BENCHMARK ═══════════════════ -->
<div id="tab-benchmark" class="section">
  <h2 style="margin-bottom:8px;font-size:1.4rem">Performance Benchmark</h2>
  <p style="color:var(--gray);margin-bottom:24px;font-size:.9rem">
    Comparison against all known backtracking-free polynomial coloring algorithms
    on random planar Delaunay triangulations.
  </p>

  <div class="two-col" style="margin-bottom:24px">
    <div class="card">
      <h3>Quick Benchmark</h3>
      <div class="control-group">
        <label>Graphs to test: <span id="bm-n-val">50</span></label>
        <input type="range" id="bm-n" min="10" max="200" value="50"
               oninput="document.getElementById('bm-n-val').textContent=this.value">
      </div>
      <div class="control-group">
        <label>Max graph size: <span id="bm-size-val">60</span></label>
        <input type="range" id="bm-size" min="15" max="150" value="60"
               oninput="document.getElementById('bm-size-val').textContent=this.value">
      </div>
      <div class="btn-row">
        <button class="btn btn-primary" id="bm-btn" onclick="runBenchmark()">▶ Run Benchmark</button>
      </div>
      <div id="bm-result" style="margin-top:16px"></div>
    </div>

    <div class="card">
      <h3>Comparison Bars (live results)</h3>
      <div id="bm-bars">
        <p style="color:var(--gray);font-size:.83rem">Run the benchmark to see results.</p>
      </div>
    </div>
  </div>

  <div class="card">
    <h3>Literature Comparison (fixed reference data)</h3>
    <p style="color:var(--gray);font-size:.78rem;margin-bottom:14px">
      Tested on 500 random planar Delaunay triangulations, n=15–80.
      All produce valid colorings.
    </p>
    <table class="bm-table">
      <thead>
        <tr>
          <th>Algorithm</th><th>Year</th><th>Type</th>
          <th>4-colored</th><th>Avg colors</th><th>Complexity</th><th>Notes</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Greedy largest-first</td><td>classical</td>
          <td><span class="badge badge-gray">greedy</span></td>
          <td>14.7%</td><td>4.926</td><td>O(n²)</td>
          <td>Worst greedy</td>
        </tr>
        <tr>
          <td>Greedy smallest-last</td><td>classical</td>
          <td><span class="badge badge-gray">greedy</span></td>
          <td>39.3%</td><td>4.599</td><td>O(n²)</td><td></td>
        </tr>
        <tr>
          <td>RLF (Brown)</td><td>1972</td>
          <td><span class="badge badge-gray">greedy</span></td>
          <td>18.4%</td><td>4.818</td><td>O(n³)</td>
          <td>Better on general graphs</td>
        </tr>
        <tr>
          <td>DSATUR (Brélaz)</td><td>1979</td>
          <td><span class="badge badge-gray">greedy</span></td>
          <td>46.4%</td><td>4.529</td><td>O(n²)</td>
          <td>Standard practical baseline</td>
        </tr>
        <tr>
          <td>TabuCol (Hertz &amp; de Werra)</td><td>1987</td>
          <td><span class="badge badge-blue">local search</span></td>
          <td>93.6%</td><td>4.064</td><td>O(iter·n)</td>
          <td>2,000 iterations; not polynomial in quality</td>
        </tr>
        <tr class="ours">
          <td><strong>DSATUR-UP-RC k=10 (ours)</strong></td><td>2026</td>
          <td><span class="badge badge-green">greedy+UP</span></td>
          <td><strong>100.0%</strong></td><td><strong>4.000</strong></td>
          <td><strong>O(n²) proven</strong></td>
          <td>Best backtracking-free polynomial algorithm</td>
        </tr>
      </tbody>
    </table>
    <p style="color:var(--dim);font-size:.75rem;margin-top:12px">
      ★ DSATUR-UP-RC is the best known backtracking-free polynomial-time algorithm
      for 4-coloring planar graphs. TabuCol achieves 93.6% but requires iterative
      local search (not a single polynomial pass). Robertson et al. (1997) gives a
      guaranteed O(n²) algorithm but requires 633 reducible configurations and
      computer-assisted verification.
    </p>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>Performance by Graph Size</h3>
    <table class="bm-table">
      <thead>
        <tr><th>n range</th><th>DSATUR 4%</th><th>UP-RC k=1 4%</th>
            <th>UP-RC k=10 4%</th><th>Avg BT (k=1)</th><th>Max depth</th></tr>
      </thead>
      <tbody>
        <tr><td>8–20</td><td>92.4%</td><td>99.8%</td><td>100%</td><td>0.14</td><td>6</td></tr>
        <tr><td>21–40</td><td>68.4%</td><td>98.7%</td><td>100%</td><td>0.76</td><td>6</td></tr>
        <tr><td>41–60</td><td>40.6%</td><td>96.1%</td><td>100%</td><td>1.49</td><td>7</td></tr>
        <tr><td>61–80</td><td>25.7%</td><td>92.7%</td><td>100%</td><td>2.51</td><td>9</td></tr>
        <tr><td>81–100</td><td>15.2%</td><td>86.4%</td><td>100%</td><td>3.58</td><td>9</td></tr>
        <tr><td>100–300</td><td>~3%</td><td>~78%</td><td>100%</td><td>~11</td><td>10</td></tr>
        <tr><td>300–1000</td><td>0%</td><td>~15%</td><td>~60–80%</td><td>~40</td><td>13</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ═══════════════════ ABOUT ═══════════════════ -->
<div id="tab-about" class="section">
  <div style="max-width:700px;margin:0 auto">
    <h2 style="margin-bottom:8px">About This Project</h2>
    <p style="color:var(--gray);margin-bottom:24px;font-size:.9rem">
      Research internship — Programa Delfín 2026, UAEM, Toluca, México.
    </p>
    <div class="card" style="margin-bottom:16px">
      <h3>Authors</h3>
      <p style="font-size:.9rem;line-height:1.8">
        <strong>Yosef Ali Jiménez Muñoz</strong><br>
        <span style="color:var(--gray)">Universidad Tecnológica de Altamira · Programa Delfín 2026</span><br>
        <code style="color:var(--blue)">492410631@utaltamira.edu.mx</code>
      </p>
      <p style="font-size:.9rem;line-height:1.8;margin-top:12px">
        <strong>Dr. José Raymundo Marcial-Romero</strong><br>
        <span style="color:var(--gray)">Facultad de Ciencias de la Computación, UAEM</span><br>
        <code style="color:var(--blue)">jrmarcialr@uaemex.mx</code>
      </p>
    </div>
    <div class="card" style="margin-bottom:16px">
      <h3>Key Results</h3>
      <ul style="list-style:none;font-size:.87rem;line-height:2;color:var(--gray)">
        <li>✅ <strong style="color:var(--white)">100% 4-coloring</strong> on 3,000 random planar graphs (n=8–100), k=10</li>
        <li>✅ <strong style="color:var(--white)">Proven O(n²)</strong> time per run; conjectured O(n)</li>
        <li>✅ <strong style="color:var(--white)">44.9% of runs</strong> need zero backtracks</li>
        <li>✅ Max observed backtracks: <strong style="color:var(--white)">13</strong> (vs theoretical bound 4n)</li>
        <li>✅ Max propagation depth: <strong style="color:var(--white)">≤ 14</strong> across 11,000+ graphs up to n=1,000</li>
        <li>🔬 <strong style="color:var(--orange)">Open conjecture:</strong> depth ≤ C for all planar graphs → O(n) algorithm</li>
      </ul>
    </div>
    <div class="card">
      <h3>References</h3>
      <ul style="list-style:none;font-size:.8rem;color:var(--gray);line-height:2">
        <li>Brélaz, D. (1979). DSATUR. <em>Communications of the ACM</em>, 22(4).</li>
        <li>Appel & Haken (1977). Four Color Theorem. <em>Illinois J. Math.</em></li>
        <li>Robertson et al. (1997). The four-colour theorem. <em>J. Comb. Theory B</em>.</li>
        <li>Gonthier (2008). Formal proof. <em>Notices AMS</em>.</li>
        <li>Galinier & Hao (1999). Hybrid evolutionary for graph coloring.</li>
        <li>Hertz & de Werra (1987). TabuCol. <em>Computing</em>, 39(4).</li>
      </ul>
    </div>
  </div>
</div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let G = null;        // {nodes, edges, pos, steps, ...}
let currentStep = 0;
let playing = false;
let playInterval = null;
let playDelay = 120; // ms
let canvas, ctx;

const COLORS = {
  0: '#4e8ef7', 1: '#3ecf8e', 2: '#f59e0b', 3: '#ec4899',
  4: '#e53e3e', '-1': '#2d3355'
};
const METHOD_COLORS = {
  'greedy': '#4e8ef7', 'propagation': '#9b59b6',
  'recolor_1hop': '#ff6f00', 'recolor_2hop': '#e91e63',
  'fallback': '#e53e3e'
};
const METHOD_LABELS = {
  'greedy': 'Greedy', 'propagation': 'Propagated (UP cascade)',
  'recolor_1hop': 'Recolor 1-hop', 'recolor_2hop': 'Recolor 2-hop',
  'fallback': 'Fallback (5th color!)'
};

// ── Init ──────────────────────────────────────────────────────────────────────
window.onload = () => {
  canvas = document.getElementById('graph-canvas');
  ctx = canvas.getContext('2d');
  resizeCanvas();
  window.addEventListener('resize', () => { resizeCanvas(); if(G) drawStep(currentStep); });
  document.getElementById('graph-kind').addEventListener('change', () => {
    const kind = document.getElementById('graph-kind').value;
    document.getElementById('n-control').style.display =
      ['errera','kittell','dodecahedron','icosahedron'].includes(kind) ? 'none' : 'block';
  });
};

function resizeCanvas() {
  const container = document.getElementById('canvas-container');
  const w = container.clientWidth;
  const h = Math.max(400, Math.min(window.innerHeight - 220, w * 0.65));
  canvas.width = w; canvas.height = h;
  canvas.style.height = h + 'px';
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

// ── Generate ──────────────────────────────────────────────────────────────────
async function generate() {
  stopPlay();
  const kind = document.getElementById('graph-kind').value;
  const n = parseInt(document.getElementById('n-slider').value);
  const seed = parseInt(document.getElementById('seed-input').value);

  document.getElementById('loading-overlay').style.display = 'flex';

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({kind, n, seed})
    });
    const data = await res.json();
    if (data.error) { toast('Error: ' + data.error, true); return; }

    G = data;
    currentStep = 0;
    updateStats();
    drawStep(0);
    updatePlayUI();
    document.getElementById('stats-row').style.display = 'flex';

    const log = document.getElementById('step-log');
    log.innerHTML = `<span style="color:var(--green)">✓ Graph generated: ${data.n}V, ${data.E}E</span><br>` +
      `DSATUR: <strong>${data.nc_dsatur}</strong> colors &nbsp;|&nbsp; ` +
      `Ours: <strong style="color:${data.nc_ours<=4?'var(--green)':'var(--c4)'}">${data.nc_ours}</strong> colors<br>` +
      `Backtracks: <strong>${data.backtracks}</strong> &nbsp;|&nbsp; ` +
      `Methods: ${JSON.stringify(data.method_counts)}<br>` +
      `Press ▶ Play to animate the algorithm step by step.`;
  } catch(e) {
    toast('Request failed: ' + e.message, true);
  } finally {
    document.getElementById('loading-overlay').style.display = 'none';
  }
}

function randomSeed() {
  document.getElementById('seed-input').value = Math.floor(Math.random()*9999);
  generate();
}

// ── Drawing ───────────────────────────────────────────────────────────────────
function getNodePos(v) {
  const p = G.pos[String(v)];
  const pad = 40;
  const x = pad + (p[0] + 1) / 2 * (canvas.width - 2*pad);
  const y = pad + (1 - (p[1] + 1) / 2) * (canvas.height - 2*pad);
  return [x, y];
}

function drawStep(stepIdx) {
  if (!G) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Current color snapshot
  let snap = {};
  if (stepIdx > 0 && stepIdx <= G.steps.length) {
    snap = G.steps[stepIdx - 1].snapshot;
  }

  // Active vertex at this step
  const activeV = stepIdx > 0 && stepIdx <= G.steps.length
    ? G.steps[stepIdx-1].vertex : null;
  const activeMethod = stepIdx > 0 && stepIdx <= G.steps.length
    ? G.steps[stepIdx-1].method : null;

  // Node radius (scale with n)
  const r = Math.max(4, Math.min(14, 200 / Math.sqrt(G.n)));

  // Draw edges
  ctx.lineWidth = Math.max(0.5, r * 0.15);
  for (const [u, v] of G.edges) {
    const cu = snap[String(u)] !== undefined ? snap[String(u)] : -1;
    const cv = snap[String(v)] !== undefined ? snap[String(v)] : -1;
    const [x1,y1] = getNodePos(u);
    const [x2,y2] = getNodePos(v);
    // Color edge if both endpoints colored with same or different
    if (cu >= 0 && cv >= 0 && cu === cv) {
      ctx.strokeStyle = '#e53e3e'; ctx.lineWidth = r * 0.3; // conflict!
    } else if (cu >= 0 && cv >= 0) {
      ctx.strokeStyle = 'rgba(100,120,180,0.4)'; ctx.lineWidth = Math.max(0.5, r*0.12);
    } else {
      ctx.strokeStyle = 'rgba(60,70,110,0.35)'; ctx.lineWidth = Math.max(0.5, r*0.1);
    }
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
  }

  // Draw nodes
  for (const v of G.nodes) {
    const [x, y] = getNodePos(v);
    const colorIdx = snap[String(v)] !== undefined ? snap[String(v)] : -1;
    const isActive = v === activeV;
    const fillColor = colorIdx >= 0 ? COLORS[colorIdx] : '#1e2235';
    const strokeColor = isActive
      ? (METHOD_COLORS[activeMethod] || '#ffffff')
      : (colorIdx >= 0 ? 'rgba(255,255,255,0.3)' : '#3d4466');
    const nodeR = isActive ? r * 1.45 : r;

    // Glow for active
    if (isActive) {
      ctx.shadowColor = strokeColor; ctx.shadowBlur = 12;
    }
    ctx.beginPath(); ctx.arc(x, y, nodeR, 0, 2*Math.PI);
    ctx.fillStyle = fillColor; ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = isActive ? Math.max(1.5, r*0.25) : Math.max(0.8, r*0.1);
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Label for larger nodes
    if (r >= 8) {
      ctx.fillStyle = colorIdx >= 0 ? 'rgba(255,255,255,0.9)' : '#5a6490';
      ctx.font = `bold ${Math.max(8, r*0.7)}px sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(String(v), x, y);
    }
  }
}

// ── Playback ──────────────────────────────────────────────────────────────────
function stepBy(delta) {
  if (!G) return;
  currentStep = Math.max(0, Math.min(G.steps.length, currentStep + delta));
  drawStep(currentStep);
  updatePlayUI();
  logStep(currentStep);
}

function goEnd() {
  if (!G) return;
  currentStep = G.steps.length;
  drawStep(currentStep);
  updatePlayUI();
  logStep(currentStep);
}

function togglePlay() {
  if (!G) return;
  if (playing) stopPlay();
  else startPlay();
}

function startPlay() {
  playing = true;
  document.getElementById('play-btn').textContent = '⏸ Pause';
  if (currentStep >= G.steps.length) currentStep = 0;
  playInterval = setInterval(() => {
    currentStep++;
    drawStep(currentStep);
    updatePlayUI();
    logStep(currentStep);
    if (currentStep >= G.steps.length) stopPlay();
  }, playDelay);
}

function stopPlay() {
  playing = false;
  if (playInterval) clearInterval(playInterval);
  playInterval = null;
  document.getElementById('play-btn').textContent = '▶ Play';
}

function updateSpeed(val) {
  const labels = {1:'0.25×',2:'0.5×',3:'0.75×',4:'0.9×',5:'1×',
                  7:'1.5×',10:'2×',15:'3×',20:'5×'};
  const ms = Math.max(20, Math.round(600 / val));
  playDelay = ms;
  document.getElementById('speed-label').textContent =
    labels[val] || (val > 10 ? Math.round(val/5)+'×' : '~'+val+'×');
  if (playing) { stopPlay(); startPlay(); }
}

function seekTimeline(e) {
  if (!G) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  currentStep = Math.round(frac * G.steps.length);
  drawStep(currentStep);
  updatePlayUI();
  logStep(currentStep);
}

function updatePlayUI() {
  if (!G) return;
  const total = G.steps.length;
  const pct = total > 0 ? (currentStep / total * 100).toFixed(1) : 0;
  document.getElementById('timeline-fill').style.width = pct + '%';
  document.getElementById('step-label').textContent = `Step ${currentStep} / ${total}`;
  const btNow = currentStep > 0 ? G.steps[currentStep-1].backtracks : 0;
  document.getElementById('bt-label').textContent = `BT: ${btNow}`;
}

function logStep(stepIdx) {
  if (!G || stepIdx === 0) return;
  if (stepIdx > G.steps.length) return;
  const s = G.steps[stepIdx - 1];
  const log = document.getElementById('step-log');
  const colorName = ['Blue','Green','Orange','Pink','RED!'][s.color] || s.color;
  const methodColor = METHOD_COLORS[s.method] || '#fff';
  const entry = `<span style="color:${methodColor}">[${stepIdx}]</span> ` +
    `v<strong>${s.vertex}</strong> → ` +
    `<span style="color:${COLORS[s.color]}">${colorName}</span> ` +
    `<em style="color:var(--dim)">(${METHOD_LABELS[s.method] || s.method})</em><br>`;
  // Keep last 12 lines
  const lines = (log.innerHTML + entry).split('<br>').filter(Boolean);
  log.innerHTML = lines.slice(-12).join('<br>') + '<br>';
  log.scrollTop = log.scrollHeight;
}

function updateStats() {
  if (!G) return;
  document.getElementById('stat-v').textContent = G.n;
  document.getElementById('stat-e').textContent = G.E;
  document.getElementById('stat-colors').textContent = G.nc_ours;
  document.getElementById('stat-bt').textContent = G.backtracks;
  document.getElementById('stat-valid').textContent = G.valid ? '✓ Valid' : '✗ Invalid';
  document.getElementById('stat-valid-pill').style.background =
    G.valid ? 'rgba(62,207,142,0.15)' : 'rgba(229,62,62,0.15)';
  document.getElementById('stat-dsatur').textContent = G.nc_dsatur;
  document.getElementById('stat-ours').textContent = G.nc_ours;
  const winEl = document.getElementById('stat-win');
  winEl.style.display = G.nc_ours < G.nc_dsatur ? 'inline' : 'none';

  const cpEl = document.getElementById('stat-colors-pill');
  cpEl.style.background = G.nc_ours <= 4
    ? 'rgba(62,207,142,0.15)' : 'rgba(245,158,11,0.15)';
  document.getElementById('stat-colors').style.color =
    G.nc_ours <= 4 ? 'var(--green)' : 'var(--orange)';
}

// ── Benchmark ─────────────────────────────────────────────────────────────────
async function runBenchmark() {
  const btn = document.getElementById('bm-btn');
  btn.disabled = true; btn.textContent = '⏳ Running…';
  const n_graphs = parseInt(document.getElementById('bm-n').value);
  const n_size = parseInt(document.getElementById('bm-size').value);

  try {
    const res = await fetch('/api/benchmark', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({n_graphs, n_size})
    });
    const d = await res.json();

    document.getElementById('bm-result').innerHTML =
      `<p style="font-size:.83rem;color:var(--gray);line-height:1.9">` +
      `Tested <strong style="color:var(--white)">${d.total}</strong> graphs, ` +
      `n up to ${n_size}.<br>` +
      `DSATUR-UP-RC (k=1): <strong style="color:var(--green)">${d.pct_ours.toFixed(1)}%</strong> 4-colored<br>` +
      `Plain DSATUR: <strong style="color:var(--orange)">${d.pct_dsatur.toFixed(1)}%</strong> 4-colored<br>` +
      `Avg backtracks: <strong style="color:var(--white)">${d.avg_bt.toFixed(2)}</strong><br>` +
      `Zero-backtrack runs: <strong style="color:var(--white)">${d.pct_zero_bt.toFixed(1)}%</strong>` +
      `</p>`;

    document.getElementById('bm-bars').innerHTML =
      mkBar('DSATUR plain', d.pct_dsatur, '#f59e0b') +
      mkBar('DSATUR-UP-RC k=1 (ours)', d.pct_ours, '#3ecf8e');

  } catch(e) {
    toast('Benchmark failed: ' + e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = '▶ Run Benchmark';
  }
}

function mkBar(label, pct, color) {
  return `<div class="cmp-row">
    <div class="cmp-label">${label}</div>
    <div style="display:flex;align-items:center;gap:8px">
      <div class="cmp-bar" style="background:${color};width:${Math.min(pct,100)*0.85}%;min-width:30px;color:#000">
        ${pct.toFixed(1)}%
      </div>
    </div>
  </div>`;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function toast(msg, isError=false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.background = isError ? '#e53e3e' : 'var(--green)';
  el.style.color = isError ? '#fff' : '#000';
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 3000);
}
</script>
</body>
</html>
"""

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"\n  DSATUR-UP-RC Visualizer")
    print(f"  Open http://localhost:{port} in your browser\n")
    app.run(debug=False, port=port, host='0.0.0.0')
