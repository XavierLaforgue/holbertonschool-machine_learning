# Machine learning from scratch (Holberton School ML specialisation)

[![CI](https://github.com/XavierLaforgue/holbertonschool-machine_learning/actions/workflows/ci.yml/badge.svg)](https://github.com/XavierLaforgue/holbertonschool-machine_learning/actions/workflows/ci.yml)

Implementations of core machine-learning algorithms **from first principles in NumPy**: decision trees, random forests, isolation trees and a first neural-network unit, plus the linear algebra, calculus and plotting groundwork they rely on. Written for the Machine Learning track (RNCP level 6) of Holberton School, 2026.

The point is to build the algorithms, not to call them: scikit-learn is used only to load reference datasets in the demo scripts.

## What's implemented

| Module | Content | Highlights |
|---|---|---|
| [`supervised_learning/decision_tree`](supervised_learning/decision_tree) | `Decision_Tree`, `Node`, `Leaf`; `Random_Forest`; `Isolation_Random_Tree` | Tree building with random or **Gini-impurity** splits (Gini criterion vectorised, no Python loops), vectorised prediction via per-leaf indicator functions, random forest of randomly split trees with majority vote, isolation trees for **outlier detection** (leaf depth) |
| [`supervised_learning/classification`](supervised_learning/classification) | `Neuron`: a single logistic-regression unit | Private attributes with getters, forward propagation (sigmoid), logistic-regression cost, evaluation. Training and multi-layer networks are the next tasks |
| [`math/linear_algebra`](math/linear_algebra) | Pure-Python and NumPy matrix operations | Shape, transpose, element-wise operations, concatenation, matrix multiplication, slicing along arbitrary axes; first by hand, then vectorised with NumPy broadcasting |
| [`math/calculus`](math/calculus) | Sums, products, derivatives, integrals | Polynomial derivative and integral, closed-form summation |
| [`math/plotting`](math/plotting) | `matplotlib` figures | Line, scatter, log scale, histogram, stacked bars, multi-panel figure |

## Results

Demo scripts in `supervised_learning/decision_tree` (fixed seeds, test-set accuracy):

| Model | Circle of clouds (synthetic) | Iris | Wine |
|---|---|---|---|
| Decision tree, Gini splits (`8-main_1.py`) | 1.00 | 0.93 | 0.94 |
| Random forest (`9-main_1.py`) | 1.00 | 0.87 | 0.94 |

## Run it

Python 3.13 (the course grades on Python 3.9 with NumPy 1.25; the code runs on both). With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
cd supervised_learning/decision_tree
uv run python 9-main_1.py        # trains and evaluates a random forest
```

Or with pip: `pip install numpy scikit-learn matplotlib`, then run any `*-main*.py` script from its own folder (each `N-main.py` exercises file `N-*.py`).

The `classification` demos expect a dataset in `supervised_learning/data/` (`Binary_Train.npz`, provided by the course, not redistributed here).

## Code quality

- Style: `pycodestyle` 2.11.1 (the version the course grades with) on every module, checked by [GitHub Actions](.github/workflows/ci.yml) along with a run of the decision-tree and random-forest demos.
- Course rules: documented modules, classes and functions; executable files with a `#!/usr/bin/env python3` shebang; NumPy only, unless a task allows more.
- Tooling declared in `pyproject.toml`: `black`, `pylint`, `pyright`, `pycodestyle`, `pydocstyle`; dependencies locked with `uv`.

## Status

In progress: the first five projects of a 32-project track (linear algebra, plotting, calculus, decision trees and random forests, then classification with neural networks, started). Next in the track: TensorFlow/Keras, optimisation, regularisation, CNNs, and later NLP and reinforcement learning.

The files follow the course's numbering: each task adds a feature to the previous file, so `N-build_decision_tree.py` contains everything up to task N.
