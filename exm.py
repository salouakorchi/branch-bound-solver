"""
============================================================
  Branch and Bound — Integer Linear Programming Solver
  Optimisation en Nombres Entiers · UMMTO 4ème année
============================================================
Auteur  : Implémentation professionnelle Python
Méthode : Branch and Bound (Séparation et Évaluation)
Dépendances : scipy, numpy (pip install scipy numpy)
============================================================
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.optimize import linprog


# ──────────────────────────────────────────────────────────
#  Structures de données
# ──────────────────────────────────────────────────────────

@dataclass
class LPResult:
    """Résultat d'une relaxation linéaire."""
    feasible: bool
    objective: float = float("-inf")
    x: Optional[np.ndarray] = None


@dataclass
class Node:
    """Nœud de l'arbre Branch and Bound."""
    node_id: int
    depth: int
    lower_bounds: np.ndarray          # bornes inférieures sur x
    upper_bounds: np.ndarray          # bornes supérieures  sur x
    parent_id: Optional[int] = None
    branch_var: Optional[int] = None  # variable sur laquelle on a branché
    branch_dir: Optional[str] = None  # "left" (≤) ou "right" (≥)
    lp_value: float = float("-inf")
    lp_solution: Optional[np.ndarray] = None
    status: str = "active"            # active | integer | infeasible | pruned


@dataclass
class BBResult:
    """Résultat global de Branch and Bound."""
    optimal_value: float
    optimal_solution: np.ndarray
    nodes_explored: int
    nodes_pruned: int
    elapsed_time: float
    tree: list[Node]


# ──────────────────────────────────────────────────────────
#  Solveur de relaxation linéaire (scipy)
# ──────────────────────────────────────────────────────────

def solve_lp(c: np.ndarray,
             A_ub: np.ndarray,
             b_ub: np.ndarray,
             lb: np.ndarray,
             ub: np.ndarray) -> LPResult:
    """
    Résout la relaxation linéaire en maximisation via scipy.linprog.

    scipy minimise, donc on passe -c pour maximiser c^T x.
    """
    n = len(c)
    bounds = [(lb[i], ub[i]) for i in range(n)]

    res = linprog(
        -c,                    # maximisation → minimisation de -c
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if res.status == 0:
        return LPResult(feasible=True, objective=-res.fun, x=res.x)
    return LPResult(feasible=False)


# ──────────────────────────────────────────────────────────
#  Classe principale : Branch and Bound
# ──────────────────────────────────────────────────────────

class BranchAndBound:
    """
    Solveur Branch and Bound pour la Programmation Linéaire en Nombres Entiers.

    Maximise  c^T x
    Sous      A_ub x ≤ b_ub
              lb ≤ x ≤ ub
              x[int_vars] ∈ ℤ

    Paramètres
    ----------
    c        : coefficients de la fonction objectif (vecteur de taille n)
    A_ub     : matrice des contraintes  (m × n)
    b_ub     : vecteur des ressources   (m,)
    int_vars : indices des variables devant être entières (None = toutes)
    lb       : bornes inférieures (défaut 0)
    ub       : bornes supérieures (défaut +∞)
    tol      : tolérance pour le test d'intégralité
    verbose  : affichage détaillé de l'exploration
    strategy : stratégie d'exploration ('depth', 'breadth', 'best')
    """

    STRATEGIES = ("depth", "breadth", "best")

    def __init__(
        self,
        c: list | np.ndarray,
        A_ub: list | np.ndarray,
        b_ub: list | np.ndarray,
        int_vars: Optional[list[int]] = None,
        lb: Optional[list | np.ndarray] = None,
        ub: Optional[list | np.ndarray] = None,
        tol: float = 1e-6,
        verbose: bool = True,
        strategy: str = "best",
    ):
        self.c = np.asarray(c, dtype=float)
        self.A_ub = np.asarray(A_ub, dtype=float)
        self.b_ub = np.asarray(b_ub, dtype=float)
        self.n = len(self.c)
        self.int_vars = set(int_vars) if int_vars is not None else set(range(self.n))
        self.tol = tol
        self.verbose = verbose

        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy doit être parmi {self.STRATEGIES}")
        self.strategy = strategy

        # Bornes globales
        self._lb_global = np.zeros(self.n) if lb is None else np.asarray(lb, dtype=float)
        self._ub_global = np.full(self.n, np.inf) if ub is None else np.asarray(ub, dtype=float)

        # Compteurs
        self._node_counter = 0
        self._nodes_pruned = 0
        self._tree: list[Node] = []

    # ── Helpers ──────────────────────────────────────────

    def _is_integer(self, x: np.ndarray) -> bool:
        return all(abs(x[i] - round(x[i])) <= self.tol for i in self.int_vars)

    def _first_fractional(self, x: np.ndarray) -> Optional[int]:
        """Renvoie l'indice de la première variable fractionnaire parmi int_vars."""
        for i in sorted(self.int_vars):
            if abs(x[i] - round(x[i])) > self.tol:
                return i
        return None

    def _new_node(self, parent_id, depth, lb, ub, branch_var=None, branch_dir=None) -> Node:
        self._node_counter += 1
        node = Node(
            node_id=self._node_counter,
            depth=depth,
            lower_bounds=lb.copy(),
            upper_bounds=ub.copy(),
            parent_id=parent_id,
            branch_var=branch_var,
            branch_dir=branch_dir,
        )
        self._tree.append(node)
        return node

    def _pop_node(self, active: list[Node]) -> Node:
        if self.strategy == "depth":
            return active.pop()           # LIFO → profondeur d'abord
        elif self.strategy == "breadth":
            return active.pop(0)          # FIFO → largeur d'abord
        else:  # best
            idx = max(range(len(active)), key=lambda i: active[i].lp_value)
            return active.pop(idx)        # meilleure borne supérieure

    # ── Affichage ────────────────────────────────────────

    def _log(self, node: Node, msg: str = ""):
        if not self.verbose:
            return
        indent = "  " * node.depth
        prefix = f"[N{node.node_id:03d}|d{node.depth}]{indent}"
        print(f"{prefix} {msg}")

    def _print_header(self):
        if not self.verbose:
            return
        w = 68
        print("=" * w)
        print("  BRANCH AND BOUND — Séparation et Évaluation".center(w))
        print("=" * w)
        print(f"  Variables      : {self.n}")
        print(f"  Var. entières  : {sorted(self.int_vars)}")
        print(f"  Contraintes    : {self.A_ub.shape[0]}")
        print(f"  Stratégie      : {self.strategy}")
        print("-" * w)

    def _print_footer(self, res: BBResult):
        if not self.verbose:
            return
        w = 68
        print("=" * w)
        print("  RÉSULTAT FINAL".center(w))
        print("=" * w)
        print(f"  Solution optimale : x* = {np.round(res.optimal_solution, 4)}")
        print(f"  Valeur optimale   : Z* = {res.optimal_value:.6f}")
        print(f"  Nœuds explorés    : {res.nodes_explored}")
        print(f"  Nœuds élagués     : {res.nodes_pruned}")
        print(f"  Temps d'exécution : {res.elapsed_time:.4f} s")
        print("=" * w)

    # ── Algorithme principal ─────────────────────────────

    def solve(self) -> BBResult:
        """Lance l'algorithme Branch and Bound et retourne le résultat."""
        t0 = time.perf_counter()
        self._print_header()

        # ── Initialisation ──
        best_value = float("-inf")   # BI (borne inférieure)
        best_solution: Optional[np.ndarray] = None

        root = self._new_node(
            parent_id=None, depth=0,
            lb=self._lb_global, ub=self._ub_global,
        )

        # Résolution de la relaxation racine
        res0 = solve_lp(self.c, self.A_ub, self.b_ub, root.lower_bounds, root.upper_bounds)
        if not res0.feasible:
            print("⚠  Le problème initial est irréalisable.")
            return BBResult(float("-inf"), np.zeros(self.n), 0, 0, 0.0, self._tree)

        root.lp_value = res0.objective
        root.lp_solution = res0.x
        root.status = "active"

        self._log(root, f"Relaxation → Z = {root.lp_value:.4f} | x = {np.round(root.lp_solution,4)}")

        active: list[Node] = [root]
        global_bs = root.lp_value     # BS globale

        nodes_explored = 0

        # ── Boucle principale ──
        while active:
            node = self._pop_node(active)
            nodes_explored += 1

            # Résolution de la relaxation de ce nœud
            lp = solve_lp(self.c, self.A_ub, self.b_ub, node.lower_bounds, node.upper_bounds)

            # ── Cas 1 : Irréalisable ──
            if not lp.feasible:
                node.status = "infeasible"
                self._nodes_pruned += 1
                self._log(node, "✗ Irréalisable → élagage")
                continue

            node.lp_value = lp.objective
            node.lp_solution = lp.x

            # ── Cas 2 : Élagage par borne ──
            if lp.objective <= best_value + self.tol:
                node.status = "pruned"
                self._nodes_pruned += 1
                self._log(node, f"✗ Z = {lp.objective:.4f} ≤ BI = {best_value:.4f} → élagage par borne")
                continue

            # ── Cas 3 : Solution entière ──
            if self._is_integer(lp.x):
                node.status = "integer"
                if lp.objective > best_value + self.tol:
                    best_value = lp.objective
                    best_solution = lp.x.copy()
                    self._log(node, f"★ Solution entière! Z = {best_value:.4f} | x = {np.round(lp.x,4)} → BI mis à jour")
                else:
                    self._log(node, f"✓ Solution entière Z = {lp.objective:.4f} (non améliorante)")
                continue

            # ── Cas 4 : Branchement ──
            frac_var = self._first_fractional(lp.x)
            frac_val = lp.x[frac_var]
            floor_val = math.floor(frac_val)
            ceil_val = math.ceil(frac_val)

            self._log(node,
                f"◆ Z = {lp.objective:.4f} | x = {np.round(lp.x,4)} | Branche sur x{frac_var+1} = {frac_val:.4f}")

            # Sous-problème gauche : x[frac_var] ≤ floor
            lb_l, ub_l = node.lower_bounds.copy(), node.upper_bounds.copy()
            ub_l[frac_var] = min(ub_l[frac_var], floor_val)
            child_l = self._new_node(node.node_id, node.depth + 1, lb_l, ub_l,
                                     branch_var=frac_var, branch_dir="left")
            lp_l = solve_lp(self.c, self.A_ub, self.b_ub, lb_l, ub_l)
            if lp_l.feasible:
                child_l.lp_value = lp_l.objective
                child_l.lp_solution = lp_l.x
                self._log(child_l, f"  └─ x{frac_var+1} ≤ {floor_val} → Z_rel = {lp_l.objective:.4f}")
                if lp_l.objective > best_value + self.tol:
                    active.append(child_l)
                else:
                    child_l.status = "pruned"
                    self._nodes_pruned += 1
                    self._log(child_l, f"     ✗ Élagué immédiatement (Z ≤ BI)")
            else:
                child_l.status = "infeasible"
                self._nodes_pruned += 1
                self._log(child_l, f"  └─ x{frac_var+1} ≤ {floor_val} → Irréalisable")

            # Sous-problème droit : x[frac_var] ≥ ceil
            lb_r, ub_r = node.lower_bounds.copy(), node.upper_bounds.copy()
            lb_r[frac_var] = max(lb_r[frac_var], ceil_val)
            child_r = self._new_node(node.node_id, node.depth + 1, lb_r, ub_r,
                                     branch_var=frac_var, branch_dir="right")
            lp_r = solve_lp(self.c, self.A_ub, self.b_ub, lb_r, ub_r)
            if lp_r.feasible:
                child_r.lp_value = lp_r.objective
                child_r.lp_solution = lp_r.x
                self._log(child_r, f"  └─ x{frac_var+1} ≥ {ceil_val} → Z_rel = {lp_r.objective:.4f}")
                if lp_r.objective > best_value + self.tol:
                    active.append(child_r)
                else:
                    child_r.status = "pruned"
                    self._nodes_pruned += 1
                    self._log(child_r, f"     ✗ Élagué immédiatement (Z ≤ BI)")
            else:
                child_r.status = "infeasible"
                self._nodes_pruned += 1
                self._log(child_r, f"  └─ x{frac_var+1} ≥ {ceil_val} → Irréalisable")

            # Mise à jour BS globale
            if active:
                global_bs = max(n.lp_value for n in active if n.lp_value > float("-inf"))

        # ── Résultat ──
        elapsed = time.perf_counter() - t0
        solution = best_solution if best_solution is not None else np.zeros(self.n)
        result = BBResult(
            optimal_value=best_value,
            optimal_solution=solution,
            nodes_explored=nodes_explored,
            nodes_pruned=self._nodes_pruned,
            elapsed_time=elapsed,
            tree=self._tree,
        )
        self._print_footer(result)
        return result

    # ── Affichage de l'arbre ─────────────────────────────

    def print_tree(self, result: BBResult):
        """Affiche un résumé tabulaire de l'arbre d'exploration."""
        STATUS_ICONS = {
            "integer":    "★ Entier",
            "infeasible": "✗ Irrél.",
            "pruned":     "⊘ Élagué",
            "active":     "◆ Actif ",
        }
        print("\n" + "─" * 80)
        print(f"{'Nœud':>5} {'Parent':>7} {'Prof.':>5} {'Branche':^14} {'Z_relax':>10} {'Statut':<12}")
        print("─" * 80)
        for node in result.tree:
            branch = ""
            if node.branch_var is not None:
                sym = "≤" if node.branch_dir == "left" else "≥"
                val = (math.floor if node.branch_dir == "left" else math.ceil)(
                    node.lp_solution[node.branch_var]
                    if node.lp_solution is not None and node.parent_id is not None
                    else 0
                )
                branch = f"x{node.branch_var+1} {sym} {val}"
            z_str = f"{node.lp_value:.4f}" if node.lp_value > float('-inf') else "—"
            icon = STATUS_ICONS.get(node.status, node.status)
            parent_str = str(node.parent_id) if node.parent_id is not None else "—"
            print(f"{node.node_id:>5} {parent_str:>7} {node.depth:>5} {branch:^14} {z_str:>10} {icon}")
        print("─" * 80)


# ──────────────────────────────────────────────────────────
#  Exemples du cours
# ──────────────────────────────────────────────────────────

def exemple_cours_1():
    """
    Exemple 1 (section 1.2 du cours) :
        max  Z  = 3x1 + 2x2
        s.c. 2x1 + x2  ≤ 4
             x1  + 2x2 ≤ 5
             x1, x2 ≥ 0, entiers
    Solution attendue : Z* = 6, (x1, x2) = (2, 0)
    """
    print("\n" + "█" * 68)
    print("  EXEMPLE 1 — Section 1.2 du cours".center(68))
    print("█" * 68)

    c    = [3, 2]
    A_ub = [[2, 1],
            [1, 2]]
    b_ub = [4, 5]

    solver = BranchAndBound(c, A_ub, b_ub, strategy="best", verbose=True)
    result = solver.solve()
    solver.print_tree(result)
    return result


def exemple_cours_2():
    """
    Exemple 2 (section 1.4 du cours) :
        max  Z  = 5x1 + 4x2
        s.c. x1  + x2   ≤ 5
             10x1 + 6x2 ≤ 45
             x1, x2 ≥ 0, entiers
    Solution attendue : Z* = 13, (x1, x2) = (3, 2)
    """
    print("\n" + "█" * 68)
    print("  EXEMPLE 2 — Section 1.4 du cours (exemple détaillé)".center(68))
    print("█" * 68)

    c    = [5, 4]
    A_ub = [[ 1,  1],
            [10,  6]]
    b_ub = [5, 45]

    solver = BranchAndBound(c, A_ub, b_ub, strategy="best", verbose=True)
    result = solver.solve()
    solver.print_tree(result)
    return result


def exemple_production():
    """
    Exemple entreprise (section 0.2 du cours) :
        max  Z  = 3x1 + 5x2
        s.c. 2x1 + 4x2 ≤ 20   (heures de travail)
             x1  + 3x2 ≤ 15   (matières premières)
             x1, x2 ≥ 0, entiers
    """
    print("\n" + "█" * 68)
    print("  EXEMPLE PRODUCTION — Section 0.3 du cours".center(68))
    print("█" * 68)

    c    = [3, 5]
    A_ub = [[2, 4],
            [1, 3]]
    b_ub = [20, 15]

    solver = BranchAndBound(c, A_ub, b_ub, strategy="best", verbose=True)
    result = solver.solve()
    solver.print_tree(result)
    return result


def exemple_mixte():
    """
    Exemple MILP : x1 entier, x2 continue
        max  Z  = 4x1 + 3x2
        s.c. 2x1 + x2  ≤ 10
             x1  + 2x2 ≤ 14
             x1 ≥ 0 entier, x2 ≥ 0 continue
    """
    print("\n" + "█" * 68)
    print("  EXEMPLE MILP — Variable mixte".center(68))
    print("█" * 68)

    c    = [4, 3]
    A_ub = [[2, 1],
            [1, 2]]
    b_ub = [10, 14]

    solver = BranchAndBound(c, A_ub, b_ub, int_vars=[0], strategy="depth", verbose=True)
    result = solver.solve()
    solver.print_tree(result)
    return result


# ──────────────────────────────────────────────────────────
#  Point d'entrée
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    exemple_cours_1()
    exemple_cours_2()
    exemple_production()
    exemple_mixte()