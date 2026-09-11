# 🔢 Branch & Bound Solver — Plateforme Interactive
### UMMTO · 4ème Année · Optimisation en Nombres Entiers

---

## 📦 Installation

```bash
# 1. Cloner / copier les fichiers dans un dossier
mkdir bb_solver && cd bb_solver
# Copiez app.py et requirements.txt ici

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement sur **http://localhost:8501**

---

## 🚀 Fonctionnalités

| Fonctionnalité | Détail |
|---|---|
| **4 exemples prédéfinis** | Exemples du cours UMMTO chargés en un clic |
| **Saisie personnalisée** | Jusqu'à 6 variables et 8 contraintes |
| **ILP & MILP** | Variables toutes entières ou choix manuel |
| **3 stratégies** | Best-First, Depth-First, Breadth-First |
| **Arbre interactif** | Visualisation Plotly de l'arbre B&B |
| **Région 2D** | Heatmap de la zone réalisable (2 variables) |
| **Console logs** | Trace complète de l'exploration nœud par nœud |
| **Export CSV / TXT** | Téléchargement de l'arbre et du rapport |

---

## 🗂 Structure des fichiers

```
bb_solver/
├── app.py              ← Application Streamlit principale
├── requirements.txt    ← Dépendances Python
└── README.md           ← Ce fichier
```

---

## 📐 Format du problème

```
max   Z  = c₁x₁ + c₂x₂ + … + cₙxₙ
s.c.  A·x ≤ b
      xᵢ ≥ 0
      xᵢ ∈ ℤ  (variables sélectionnées)
```

---

## 🔧 Dépendances

- `streamlit` — Interface web
- `scipy` — Solveur LP (HiGHS via `linprog`)
- `numpy` — Calcul matriciel
- `pandas` — Affichage tabulaire
- `plotly` — Visualisations interactives