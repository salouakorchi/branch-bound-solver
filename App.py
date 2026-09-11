"""
============================================================
  Branch and Bound — Plateforme Interactive Streamlit
  Optimisation en Nombres Entiers · UMMTO 4ème année
============================================================
"""

import streamlit as st
import math
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import linprog

# ──────────────────────────────────────────────────────────
#  Config page
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Branch & Bound Solver",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
#  CSS — version corrigée (pas de sélecteurs agressifs)
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0c1a3a 0%, #0f2d4a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 .3rem 0;
}
.hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .8rem;
    color: #06b6d4;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: .5rem;
}
.hero-desc {
    color: #94a3b8;
    font-size: .9rem;
    margin: .5rem 0 .8rem 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,.15);
    border: 1px solid rgba(59,130,246,.35);
    color: #3b82f6;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .7rem;
    padding: .22rem .65rem;
    border-radius: 4px;
}

/* KPI boxes */
.kpi-box {
    background: #161b27;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.1rem 1rem;
    text-align: center;
}
.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #3b82f6;
    line-height: 1.1;
}
.kpi-val.green { color: #10b981; }
.kpi-val.amber { color: #f59e0b; }
.kpi-val.cyan  { color: #06b6d4; }
.kpi-lbl {
    font-size: .68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: .3rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* Solution result */
.result-box {
    background: linear-gradient(135deg, rgba(16,185,129,.07), rgba(6,182,212,.07));
    border: 1px solid rgba(16,185,129,.3);
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin: 1rem 0;
    text-align: center;
}
.result-z {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #10b981;
    line-height: 1.1;
}
.result-lbl {
    font-size: .72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: .4rem;
    font-family: 'IBM Plex Mono', monospace;
}
.result-vars {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    color: #06b6d4;
    margin-top: .7rem;
}

/* Log console */
.log-box {
    background: #060a12;
    border: 1px solid #1a2540;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .76rem;
    color: #94a3b8;
    max-height: 360px;
    overflow-y: auto;
    line-height: 1.85;
}
.log-star   { color: #10b981; font-weight: 600; }
.log-prune  { color: #f59e0b; }
.log-branch { color: #3b82f6; }
.log-infeas { color: #ef4444; }
.log-root   { color: #06b6d4; }

/* Section separator */
.sep { border-top: 1px solid #1e293b; margin: 1.2rem 0; }

/* Example info card */
.ex-card {
    background: rgba(6,182,212,.07);
    border: 1px solid rgba(6,182,212,.2);
    border-radius: 8px;
    padding: .85rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .74rem;
    color: #06b6d4;
    line-height: 1.7;
    margin-top: .8rem;
}
.ex-card b { color: #e2e8f0; }
.ex-ok { color: #10b981; }

/* constraint row label */
.leq {
    padding-top: 2rem;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  Structures de données & Solveur
# ──────────────────────────────────────────────────────────

@dataclass
class LPResult:
    feasible: bool
    objective: float = float("-inf")
    x: Optional[np.ndarray] = None


@dataclass
class Node:
    node_id: int
    depth: int
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    parent_id: Optional[int] = None
    branch_var: Optional[int] = None
    branch_dir: Optional[str] = None
    lp_value: float = float("-inf")
    lp_solution: Optional[np.ndarray] = None
    status: str = "active"


@dataclass
class BBResult:
    optimal_value: float
    optimal_solution: np.ndarray
    nodes_explored: int
    nodes_pruned: int
    elapsed_time: float
    tree: list
    logs: list


def solve_lp(c, A_ub, b_ub, lb, ub):
    n = len(c)
    bounds = [(lb[i], ub[i]) for i in range(n)]
    res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status == 0:
        return LPResult(feasible=True, objective=-res.fun, x=res.x)
    return LPResult(feasible=False)


class BranchAndBound:
    def __init__(self, c, A_ub, b_ub, int_vars=None, lb=None, ub=None,
                 tol=1e-6, strategy="best"):
        self.c      = np.asarray(c, dtype=float)
        self.A_ub   = np.asarray(A_ub, dtype=float)
        self.b_ub   = np.asarray(b_ub, dtype=float)
        self.n      = len(self.c)
        self.int_vars = set(int_vars) if int_vars is not None else set(range(self.n))
        self.tol      = tol
        self.strategy = strategy
        self._lb_global = np.zeros(self.n) if lb is None else np.asarray(lb, dtype=float)
        self._ub_global = np.full(self.n, np.inf) if ub is None else np.asarray(ub, dtype=float)
        self._node_counter = 0
        self._nodes_pruned = 0
        self._tree = []
        self._logs = []

    def _log(self, msg, kind="info"):
        self._logs.append({"msg": msg, "kind": kind})

    def _is_integer(self, x):
        return all(abs(x[i] - round(x[i])) <= self.tol for i in self.int_vars)

    def _first_fractional(self, x):
        for i in sorted(self.int_vars):
            if abs(x[i] - round(x[i])) > self.tol:
                return i
        return None

    def _new_node(self, parent_id, depth, lb, ub, branch_var=None, branch_dir=None):
        self._node_counter += 1
        node = Node(
            node_id=self._node_counter, depth=depth,
            lower_bounds=lb.copy(), upper_bounds=ub.copy(),
            parent_id=parent_id, branch_var=branch_var, branch_dir=branch_dir,
        )
        self._tree.append(node)
        return node

    def _pop_node(self, active):
        if self.strategy == "depth":
            return active.pop()
        elif self.strategy == "breadth":
            return active.pop(0)
        else:
            idx = max(range(len(active)), key=lambda i: active[i].lp_value)
            return active.pop(idx)

    def solve(self):
        t0 = time.perf_counter()
        best_value    = float("-inf")
        best_solution = None

        root = self._new_node(None, 0, self._lb_global, self._ub_global)
        res0 = solve_lp(self.c, self.A_ub, self.b_ub, root.lower_bounds, root.upper_bounds)

        if not res0.feasible:
            self._log("Probleme initial irrealisable.", "infeas")
            return BBResult(float("-inf"), np.zeros(self.n), 0, 0, 0.0, self._tree, self._logs)

        root.lp_value    = res0.objective
        root.lp_solution = res0.x
        self._log(
            f"[N001 | racine] Relaxation LP => Z = {res0.objective:.4f} | x = {np.round(res0.x, 4)}",
            "root")

        active         = [root]
        nodes_explored = 0

        while active:
            node = self._pop_node(active)
            nodes_explored += 1

            lp = solve_lp(self.c, self.A_ub, self.b_ub, node.lower_bounds, node.upper_bounds)

            if not lp.feasible:
                node.status = "infeasible"
                self._nodes_pruned += 1
                self._log(f"[N{node.node_id:03d} | d{node.depth}] X Irrealisable => elagage", "infeas")
                continue

            node.lp_value    = lp.objective
            node.lp_solution = lp.x

            if lp.objective <= best_value + self.tol:
                node.status = "pruned"
                self._nodes_pruned += 1
                self._log(
                    f"[N{node.node_id:03d} | d{node.depth}] Z = {lp.objective:.4f} <= BI = {best_value:.4f} => elagage",
                    "prune")
                continue

            if self._is_integer(lp.x):
                node.status = "integer"
                if lp.objective > best_value + self.tol:
                    best_value    = lp.objective
                    best_solution = lp.x.copy()
                    self._log(
                        f"[N{node.node_id:03d} | d{node.depth}] ** Solution entiere! Z* = {best_value:.4f} | x* = {np.round(lp.x, 4)}",
                        "star")
                continue

            frac_var  = self._first_fractional(lp.x)
            frac_val  = lp.x[frac_var]
            floor_val = math.floor(frac_val)
            ceil_val  = math.ceil(frac_val)

            self._log(
                f"[N{node.node_id:03d} | d{node.depth}] Branche sur x{frac_var+1} = {frac_val:.4f}",
                "branch")

            for direction, new_bound in [("left", floor_val), ("right", ceil_val)]:
                lb_new = node.lower_bounds.copy()
                ub_new = node.upper_bounds.copy()
                if direction == "left":
                    ub_new[frac_var] = min(ub_new[frac_var], new_bound)
                else:
                    lb_new[frac_var] = max(lb_new[frac_var], new_bound)

                child = self._new_node(node.node_id, node.depth + 1, lb_new, ub_new,
                                       branch_var=frac_var, branch_dir=direction)
                lp_c  = solve_lp(self.c, self.A_ub, self.b_ub, lb_new, ub_new)
                sym   = "<=" if direction == "left" else ">="

                if lp_c.feasible:
                    child.lp_value    = lp_c.objective
                    child.lp_solution = lp_c.x
                    self._log(
                        f"  [N{child.node_id:03d}] x{frac_var+1} {sym} {new_bound} => Z_rel = {lp_c.objective:.4f}",
                        "branch")
                    if lp_c.objective > best_value + self.tol:
                        active.append(child)
                    else:
                        child.status = "pruned"
                        self._nodes_pruned += 1
                        self._log(f"     X Elague immediatement (Z <= BI)", "prune")
                else:
                    child.status = "infeasible"
                    self._nodes_pruned += 1
                    self._log(f"  [N{child.node_id:03d}] x{frac_var+1} {sym} {new_bound} => Irrealisable", "infeas")

        elapsed   = time.perf_counter() - t0
        solution  = best_solution if best_solution is not None else np.zeros(self.n)
        return BBResult(
            optimal_value    = best_value,
            optimal_solution = solution,
            nodes_explored   = nodes_explored,
            nodes_pruned     = self._nodes_pruned,
            elapsed_time     = elapsed,
            tree             = self._tree,
            logs             = self._logs,
        )


# ──────────────────────────────────────────────────────────
#  Exemples prédéfinis
# ──────────────────────────────────────────────────────────

EXAMPLES = {
    "Exemple 1 — Section 1.2": {
        "desc":     "max 3x1 + 2x2  s.c. 2x1+x2<=4  x1+2x2<=5",
        "c":        [3, 2], "n_vars": 2,
        "A":        [[2, 1], [1, 2]], "b": [4, 5],
        "int_vars": "all",
        "expected": "Z* = 6  ·  x* = (2, 0)",
    },
    "Exemple 2 — Section 1.4": {
        "desc":     "max 5x1 + 4x2  s.c. x1+x2<=5  10x1+6x2<=45",
        "c":        [5, 4], "n_vars": 2,
        "A":        [[1, 1], [10, 6]], "b": [5, 45],
        "int_vars": "all",
        "expected": "Z* = 13  ·  x* = (3, 2)",
    },
    "Exemple Production — Section 0.3": {
        "desc":     "max 3x1 + 5x2  s.c. 2x1+4x2<=20  x1+3x2<=15",
        "c":        [3, 5], "n_vars": 2,
        "A":        [[2, 4], [1, 3]], "b": [20, 15],
        "int_vars": "all",
        "expected": "Z* = 25  ·  x* = (0, 5)",
    },
    "Exemple MILP — Mixte": {
        "desc":     "max 4x1 + 3x2  s.c. 2x1+x2<=10  x1+2x2<=14  (x1 entier)",
        "c":        [4, 3], "n_vars": 2,
        "A":        [[2, 1], [1, 2]], "b": [10, 14],
        "int_vars": [0],
        "expected": "x1 entier, x2 continu",
    },
}

# ──────────────────────────────────────────────────────────
#  Graphiques Plotly
# ──────────────────────────────────────────────────────────

CLRS = {
    "integer":    "#10b981",
    "pruned":     "#f59e0b",
    "infeasible": "#ef4444",
    "active":     "#3b82f6",
}

def fig_tree(tree):
    if not tree:
        return None

    levels = {}
    for n in tree:
        levels.setdefault(n.depth, []).append(n)

    pos = {}
    for depth, nodes in sorted(levels.items()):
        w = len(nodes)
        for i, n in enumerate(nodes):
            pos[n.node_id] = ((i - (w - 1) / 2) * 2.8, -depth * 2.0)

    ex, ey = [], []
    for n in tree:
        if n.parent_id and n.parent_id in pos:
            x0, y0 = pos[n.parent_id]
            x1, y1 = pos[n.node_id]
            ex += [x0, x1, None]
            ey += [y0, y1, None]

    nx  = [pos[n.node_id][0] for n in tree]
    ny  = [pos[n.node_id][1] for n in tree]
    clr = [CLRS.get(n.status, "#64748b") for n in tree]
    tip = [
        f"<b>N{n.node_id}</b><br>Profondeur : {n.depth}<br>"
        f"Z relaxe : {n.lp_value:.4f}<br>Statut : {n.status}<br>"
        f"Sol : {np.round(n.lp_solution, 3) if n.lp_solution is not None else '---'}"
        for n in tree
    ]

    f = go.Figure()
    f.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                           line=dict(color="#1e293b", width=2),
                           hoverinfo="skip", showlegend=False))
    f.add_trace(go.Scatter(x=nx, y=ny, mode="markers+text",
                           marker=dict(color=clr, size=40,
                                       line=dict(color="#0f1117", width=2)),
                           text=[f"N{n.node_id}" for n in tree],
                           textfont=dict(color="white", size=11,
                                         family="IBM Plex Mono"),
                           textposition="middle center",
                           hovertext=tip, hoverinfo="text",
                           showlegend=False))
    for status, color in CLRS.items():
        f.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                               marker=dict(color=color, size=11),
                               name=status.capitalize()))
    f.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(family="IBM Plex Mono", color="#94a3b8"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(22,27,39,.9)", bordercolor="#1e293b",
                    borderwidth=1),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=440,
    )
    return f


def fig_bars(tree):
    rows = [{"Noeud": f"N{n.node_id}", "Z": n.lp_value,
             "Profondeur": n.depth, "Statut": n.status}
            for n in tree if n.lp_value > float("-inf")]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    f = px.bar(df, x="Noeud", y="Z", color="Statut",
               color_discrete_map=CLRS, hover_data=["Profondeur"],
               labels={"Z": "Valeur LP relaxee"})
    f.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(family="IBM Plex Mono", color="#94a3b8"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(22,27,39,.9)", bordercolor="#1e293b",
                    borderwidth=1),
        margin=dict(l=40, r=10, t=10, b=40),
        height=330, bargap=0.3,
    )
    return f


def fig_2d(c, A_ub, b_ub, result):
    if len(c) != 2:
        return None

    x_max = 0
    for row, rhs in zip(A_ub, b_ub):
        if row[0] > 0: x_max = max(x_max, rhs / row[0])
        if row[1] > 0: x_max = max(x_max, rhs / row[1])
    x_max = min(x_max * 1.25, 60)

    N  = 350
    xs = np.linspace(0, x_max, N)
    ys = np.linspace(0, x_max, N)
    XX, YY = np.meshgrid(xs, ys)
    feas = np.ones_like(XX, dtype=bool)
    for row, rhs in zip(A_ub, b_ub):
        feas &= (row[0] * XX + row[1] * YY <= rhs + 1e-9)
    feas &= (XX >= 0) & (YY >= 0)

    Z_obj = np.where(feas, c[0] * XX + c[1] * YY, np.nan)

    f = go.Figure()
    f.add_trace(go.Contour(
        x=xs, y=ys, z=Z_obj,
        colorscale=[[0, "#0c1a3a"], [0.5, "#1d4ed8"], [1, "#06b6d4"]],
        showscale=True, opacity=0.5,
        contours=dict(showlines=True, coloring="heatmap"),
        colorbar=dict(title="Z(x)", tickfont=dict(color="#94a3b8"),
                      bgcolor="rgba(22,27,39,.8)", bordercolor="#1e293b"),
    ))

    for i, (row, rhs) in enumerate(zip(A_ub, b_ub)):
        if abs(row[1]) > 1e-9:
            y_line = np.clip((rhs - row[0] * xs) / row[1], 0, x_max)
            f.add_trace(go.Scatter(x=xs, y=y_line, mode="lines",
                                   line=dict(color="#f59e0b", width=1.5, dash="dash"),
                                   name=f"C{i+1}"))

    if result and result.optimal_value > float("-inf"):
        xo = result.optimal_solution
        f.add_trace(go.Scatter(
            x=[xo[0]], y=[xo[1]], mode="markers",
            marker=dict(color="#10b981", size=14, symbol="star",
                        line=dict(color="white", width=2)),
            name=f"x* = ({xo[0]:.2f}, {xo[1]:.2f})"))

    f.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font=dict(family="IBM Plex Mono", color="#94a3b8"),
        xaxis=dict(title="x1", showgrid=True, gridcolor="#1e293b", range=[0, x_max]),
        yaxis=dict(title="x2", showgrid=True, gridcolor="#1e293b", range=[0, x_max]),
        legend=dict(bgcolor="rgba(22,27,39,.9)", bordercolor="#1e293b", borderwidth=1),
        margin=dict(l=40, r=10, t=10, b=40),
        height=380,
    )
    return f


# ──────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:.4rem 0 1rem 0;">
        <div style="font-size:2rem;">🔢</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:.68rem;
                    color:#06b6d4;text-transform:uppercase;letter-spacing:.12em;">
            B&B Solver · UMMTO
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Configuration")

    example_choice = st.selectbox(
        "Exemple prédéfini",
        ["— Personnalise —"] + list(EXAMPLES.keys()),
    )
    use_example = example_choice != "— Personnalise —"
    ex = EXAMPLES.get(example_choice, {})

    st.markdown("---")
    st.markdown("#### Dimensions")

    n_vars = st.number_input("Nombre de variables (n)", 2, 6,
                              value=int(ex.get("n_vars", 2)), step=1,
                              disabled=use_example)
    n_cons = st.number_input("Nombre de contraintes (m)", 1, 8,
                              value=int(len(ex.get("A", [[0]] * 2))), step=1,
                              disabled=use_example)

    st.markdown("---")
    st.markdown("#### Parametres")

    strategy = st.selectbox(
        "Strategie",
        ["best", "depth", "breadth"],
        format_func=lambda s: {
            "best":    "Best-First (recommande)",
            "depth":   "Depth-First",
            "breadth": "Breadth-First",
        }[s],
        disabled=use_example,
    )
    tol = st.select_slider(
        "Tolerance",
        options=[1e-3, 1e-4, 1e-5, 1e-6],
        value=1e-6,
        format_func=lambda v: f"{v:.0e}",
    )

    st.markdown("---")
    st.markdown("#### Variables entieres")
    integrality = st.radio(
        "Integralite",
        ["Toutes entieres (ILP)", "Selection manuelle (MILP)"],
        disabled=use_example,
    )


# ──────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-sub">Optimisation en Nombres Entiers</div>
    <div class="hero-title">Branch &amp; Bound Solver</div>
    <div class="hero-desc">
        Resolution interactive de problemes ILP &amp; MILP avec visualisation
        complete de l'arbre d'exploration.
    </div>
    <span class="hero-badge">UMMTO &nbsp;·&nbsp; 4eme Annee &nbsp;·&nbsp; Separation &amp; Evaluation</span>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  FORMULAIRE
# ──────────────────────────────────────────────────────────

col_form, col_btn = st.columns([4, 1])

with col_form:
    with st.expander("Definir le probleme", expanded=True):
        tab_c, tab_A, tab_iv = st.tabs([
            "Fonction objectif",
            "Contraintes  (Ax <= b)",
            "Variables entieres",
        ])

        # — Objectif —
        with tab_c:
            st.caption("Coefficients c :  max Z = c1*x1 + c2*x2 + ...")
            if use_example:
                c_vals = list(ex["c"])
                cols   = st.columns(len(c_vals))
                for i, v in enumerate(c_vals):
                    cols[i].number_input(f"c{i+1}", value=float(v),
                                         key=f"c_{i}", disabled=True)
            else:
                nv   = int(n_vars)
                cols = st.columns(nv)
                c_vals = []
                for i in range(nv):
                    v = cols[i].number_input(f"c{i+1}", value=1.0,
                                             step=1.0, key=f"c_{i}")
                    c_vals.append(v)

        # — Contraintes —
        with tab_A:
            st.caption("Matrice A et vecteur b — une ligne par contrainte.")
            nv = len(c_vals) if use_example else int(n_vars)
            nc = len(ex.get("A", [])) if use_example else int(n_cons)

            if use_example:
                A_vals = [list(r) for r in ex["A"]]
                b_vals = list(ex["b"])
                for i, (row, bi) in enumerate(zip(A_vals, b_vals)):
                    rcols = st.columns(nv + 2)
                    for j, aij in enumerate(row):
                        rcols[j].number_input(f"a{i+1},{j+1}", value=float(aij),
                                              key=f"a_{i}_{j}", disabled=True)
                    rcols[-2].markdown('<div class="leq"><=</div>', unsafe_allow_html=True)
                    rcols[-1].number_input(f"b{i+1}", value=float(bi),
                                           key=f"b_{i}", disabled=True)
            else:
                A_vals, b_vals = [], []
                for i in range(nc):
                    rcols = st.columns(nv + 2)
                    row = []
                    for j in range(nv):
                        v = rcols[j].number_input(f"a{i+1},{j+1}", value=1.0,
                                                   step=1.0, key=f"a_{i}_{j}")
                        row.append(v)
                    rcols[-2].markdown('<div class="leq"><=</div>', unsafe_allow_html=True)
                    bv = rcols[-1].number_input(f"b{i+1}", value=10.0,
                                                step=1.0, key=f"b_{i}")
                    A_vals.append(row)
                    b_vals.append(bv)

        # — Variables entières —
        with tab_iv:
            nv = len(c_vals) if use_example else int(n_vars)
            if use_example:
                iv_raw = ex.get("int_vars", "all")
                int_idx = list(range(nv)) if iv_raw == "all" else list(iv_raw)
                st.caption("Variables entieres pour cet exemple :")
                for i in range(nv):
                    st.checkbox(f"x{i+1} entier", value=(i in int_idx),
                                key=f"iv_{i}", disabled=True)
            elif integrality == "Toutes entieres (ILP)":
                int_idx = list(range(nv))
                st.caption("Toutes les variables sont entieres.")
                for i in range(nv):
                    st.checkbox(f"x{i+1} entier", value=True,
                                key=f"iv_{i}", disabled=True)
            else:
                int_idx = []
                iv_cols = st.columns(nv)
                for i in range(nv):
                    with iv_cols[i]:
                        if st.checkbox(f"x{i+1} entier", value=True, key=f"iv_{i}"):
                            int_idx.append(i)

with col_btn:
    st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)
    run = st.button("Resoudre", type="primary", use_container_width=True)
    if use_example:
        st.markdown(f"""
        <div class="ex-card">
            <b>Exemple charge</b><br>
            {ex.get('desc','')}<br><br>
            <span class="ex-ok">Attendu :</span><br>
            {ex.get('expected','')}
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  RÉSOLUTION
# ──────────────────────────────────────────────────────────

if run:
    c_np = np.array(c_vals, dtype=float)
    A_np = np.array(A_vals, dtype=float)
    b_np = np.array(b_vals, dtype=float)
    iv   = int_idx if int_idx else list(range(len(c_vals)))

    if A_np.ndim < 2 or A_np.shape[1] != len(c_np):
        st.error("Le nombre de colonnes de A ne correspond pas au nombre de variables.")
        st.stop()

    with st.spinner("Resolution Branch & Bound en cours..."):
        solver = BranchAndBound(c=c_np, A_ub=A_np, b_ub=b_np,
                                int_vars=iv, strategy=strategy, tol=tol)
        result = solver.solve()

    st.session_state["result"] = result
    st.session_state.update({"c": c_np, "A": A_np, "b": b_np, "iv": iv})


# ──────────────────────────────────────────────────────────
#  AFFICHAGE DES RÉSULTATS
# ──────────────────────────────────────────────────────────

if "result" in st.session_state:
    result = st.session_state["result"]
    c_np   = st.session_state["c"]
    A_np   = st.session_state["A"]
    b_np   = st.session_state["b"]
    iv     = st.session_state["iv"]
    ok     = result.optimal_value > float("-inf")

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    # — Solution —
    if ok:
        vars_str = "  ·  ".join(
            f"x{i+1}* = {v:.4f}" for i, v in enumerate(result.optimal_solution)
        )
        st.markdown(f"""
        <div class="result-box">
            <div class="result-lbl">Valeur optimale Z*</div>
            <div class="result-z">{result.optimal_value:.4f}</div>
            <div class="result-vars">{vars_str}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Aucune solution entiere realisable trouvee.")

    # — KPIs —
    k1, k2, k3, k4 = st.columns(4)
    for col, val, lbl, cls in [
        (k1, result.nodes_explored,                "Noeuds explores",  ""),
        (k2, result.nodes_pruned,                  "Noeuds elagues",   "amber"),
        (k3, len(result.tree),                     "Noeuds total",     "cyan"),
        (k4, f"{result.elapsed_time*1000:.1f} ms", "Temps CPU",        "green"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-val {cls}">{val}</div>
                <div class="kpi-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

    # — Onglets —
    t1, t2, t3, t4 = st.tabs([
        "Arbre B&B",
        "Valeurs LP par noeud",
        "Region realisable 2D",
        "Table des noeuds",
    ])

    with t1:
        f = fig_tree(result.tree)
        if f:
            st.plotly_chart(f, use_container_width=True)

    with t2:
        f = fig_bars(result.tree)
        if f:
            st.plotly_chart(f, use_container_width=True)

    with t3:
        if len(c_np) == 2:
            f = fig_2d(c_np, A_np, b_np, result)
            if f:
                st.plotly_chart(f, use_container_width=True)
        else:
            st.info("Visualisation 2D disponible uniquement pour n = 2 variables.")

    with t4:
        ICONS = {"integer": "Entier", "infeasible": "Irrealisable",
                 "pruned": "Elague", "active": "Actif"}
        rows = []
        for n in result.tree:
            z   = f"{n.lp_value:.4f}" if n.lp_value > float("-inf") else "—"
            br  = ""
            if n.branch_var is not None:
                sym = "<=" if n.branch_dir == "left" else ">="
                bv  = (math.floor if n.branch_dir == "left" else math.ceil)(
                    n.lp_solution[n.branch_var]
                    if n.lp_solution is not None else 0
                )
                br = f"x{n.branch_var+1} {sym} {bv}"
            rows.append({
                "Noeud":   f"N{n.node_id:03d}",
                "Parent": f"N{n.parent_id:03d}" if n.parent_id else "—",
                "Prof.":  n.depth,
                "Branche": br,
                "Z relaxe": z,
                "Sol. LP": str(np.round(n.lp_solution, 3)) if n.lp_solution is not None else "—",
                "Statut":  ICONS.get(n.status, n.status),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # — Console logs —
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    with st.expander("Console d'exploration (logs detailles)", expanded=False):
        html = "<div class='log-box'>"
        for e in result.logs:
            msg = e["msg"]
            cls = {"star": "log-star", "prune": "log-prune",
                   "branch": "log-branch", "infeas": "log-infeas",
                   "root": "log-root"}.get(e["kind"], "")
            html += f'<span class="{cls}">{msg}</span><br>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    # — Export —
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "Exporter l'arbre (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="bb_arbre.csv", mime="text/csv",
            use_container_width=True,
        )
    with ec2:
        if ok:
            lines = [
                "=" * 55,
                "  RAPPORT — Branch and Bound",
                "=" * 55,
                f"Z* = {result.optimal_value:.6f}",
                f"x* = {np.round(result.optimal_solution, 6)}",
                f"Variables entieres : {sorted(iv)}",
                f"Noeuds explores    : {result.nodes_explored}",
                f"Noeuds elagues     : {result.nodes_pruned}",
                f"Noeuds total       : {len(result.tree)}",
                f"Temps CPU          : {result.elapsed_time*1000:.2f} ms",
                f"Strategie          : {strategy}",
                "", "ARBRE", "-" * 55,
            ] + [
                f"{r['Noeud']} | d={r['Prof.']} | {r['Branche']:10s} | Z={r['Z relaxe']} | {r['Statut']}"
                for r in rows
            ] + ["", "LOGS", "-" * 55] + [e["msg"] for e in result.logs]

            st.download_button(
                "Rapport complet (TXT)",
                data="\n".join(lines).encode("utf-8"),
                file_name="bb_rapport.txt", mime="text/plain",
                use_container_width=True,
            )

# ──────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;border-top:1px solid #1e293b;padding-top:1rem;
            text-align:center;font-family:'IBM Plex Mono',monospace;
            font-size:.68rem;color:#334155;letter-spacing:.06em;">
    BRANCH &amp; BOUND SOLVER &nbsp;·&nbsp; UMMTO &nbsp;·&nbsp;
    4eme Annee Optimisation &nbsp;·&nbsp; Python · Streamlit · Plotly · SciPy
</div>
""", unsafe_allow_html=True)