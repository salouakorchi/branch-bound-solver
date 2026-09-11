# 🔢 Branch & Bound Solver

> **Optimization · Integer Programming · Python · Streamlit**

Application interactive développée en Python pour résoudre et visualiser des problèmes d’optimisation en nombres entiers (**ILP / MILP**) avec l’algorithme **Branch & Bound**.

## 🎯 Fonctionnalités

* Résolution de problèmes **ILP et MILP**
* Saisie de problèmes personnalisés
* Exemples prédéfinis
* Trois stratégies d’exploration :
  **Best-First, Depth-First et Breadth-First**
* Visualisation interactive de l’arbre Branch & Bound avec **Plotly**
* Visualisation 2D de la région réalisable
* Suivi de l’exploration des nœuds
* Export des résultats en **CSV / TXT**

## 🧠 Principe

```text
Problème
   ↓
Relaxation linéaire
   ↓
Branching
   ↓
Calcul des bornes
   ↓
Pruning
   ↓
Solution optimale
```

Le solveur permet ainsi d'observer le fonctionnement de Branch & Bound étape par étape.

## 🛠️ Technologies

**Python · Streamlit · SciPy · NumPy · Pandas · Plotly**

## 📁 Structure

```text
branch-bound-solver/
├── App.py
├── exm.py
├── requirements.txt
└── README.md
```

## ▶️ Installation

```bash
git clone https://github.com/salouakorchi/branch-bound-solver.git
cd branch-bound-solver
pip install -r requirements.txt
streamlit run App.py
```

L’application sera disponible sur :

```text
http://localhost:8501
```

## 🎓 Contexte

Projet universitaire réalisé à l’**Université Mouloud Mammeri de Tizi-Ouzou (UMMTO)** .
