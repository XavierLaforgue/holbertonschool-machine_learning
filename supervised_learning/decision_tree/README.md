# Decision trees, random forests and isolation forests

Built step by step; each file extends the previous one.
The statement of each task is in `N-task_description.md`.

| Files | Feature |
|---|---|
| `0`–`1-build_decision_tree.py` | `Node` / `Leaf` / `Decision_Tree` structure: depth, node and leaf counts |
| `2-build_decision_tree.py` | Tree printing |
| `3`–`4-build_decision_tree.py` | Leaf listing, per-node feature bounds |
| `5`–`6-build_decision_tree.py` | Indicator functions per leaf and vectorised prediction |
| `7-build_decision_tree.py` | Training with random split criteria; accuracy |
| `8-build_decision_tree.py` | Training with the **Gini impurity** split criterion (vectorised, no Python loops) |
| `9-random_forest.py` | `Random_Forest`: many randomly split trees, majority-vote prediction |
| `10-isolation_tree.py` | `Isolation_Random_Tree`: outlier detection by leaf depth (outliers end up alone in shallow leaves) |

Run from this folder: `python 8-main_1.py` (Gini tree) or `python 9-main_1.py` (random forest) train and evaluate on a synthetic "circle of clouds" dataset and on the iris and wine datasets.

Not done yet: task 11, the `Isolation_Random_Forest` that averages isolation trees to rank suspects.

## Background reading (from the course)

- Rokach & Maimon (2005), *Top-down induction of decision trees classifiers: a survey*
- Ho (1995), *Random decision forests*
- Liu, Ting & Zhou (2008), *Isolation forest*
