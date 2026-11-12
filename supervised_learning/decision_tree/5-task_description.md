Consider the indicator function for a given node, denoted as "n." This function is defined as follows:

* It takes a 2D NumPy array, denoted as `A`, of shape` (n_individuals, n_features)`. 
* The output of the indicator function is a 1D NumPy array, of size equals to the number of individuals (`n_individuals`), containing boolean values.
* The `i`-th element of this output array is set to `True` if the corresponding `i`-th individual meets the conditions specified by the node "n"; otherwise, it is set to `False`.

**Task:**
Write a  method `Node.update_indicator` that computes the indicator function from the `Node.lower` and `Node.upper` dictionaries  and stores it in an attribute `Node.indicator` :

* Fill in `def update_indicator(self):` in class `Node`:

```
def update_indicator(self) :

        def is_large_enough(x):
				
				#<- fill the gap : this function returns a 1D numpy array of size 
				#`n_individuals` so that the `i`-th element of the later is `True` 
				# if the `i`-th individual has all its features > the lower bounds
        
        def is_small_enough(x):
				
				#<- fill the gap : this function returns a 1D numpy array of size 
				#`n_individuals` so that the `i`-th element of the later is `True` 
				# if the `i`-th individual has all its features <= the lower bounds
        
        self.indicator = lambda x : np.all(np.array([is_large_enough(x),is_small_enough(x)]),axis=0)
```

**Hint:** you might want to consider something like `np.array([np.greater(A[:,key],self.lower[key]) for key in list(self.lower.keys())]` at some point.

<b> Main to test your work </b>

```
$ cat 5-main.py
#!/usr/bin/env python3

Node = __import__('5-build_decision_tree').Node
Leaf = __import__('5-build_decision_tree').Leaf
Decision_Tree = __import__('5-build_decision_tree').Decision_Tree
import numpy as np

def example_0():
    leaf0 = Leaf(0, depth=1)
    leaf1 = Leaf(0, depth=2)
    leaf2 = Leaf(1, depth=2)
    internal_node = Node(feature=1, threshold=30000, left_child=leaf1, right_child=leaf2, depth=1)
    root = Node(feature=0, threshold=.5, left_child=leaf0, right_child=internal_node, depth=0, is_root=True)
    return Decision_Tree(root=root)


def example_1(depth):
    level = [Leaf(i, depth=depth) for i in range(2 ** depth)]
    level.reverse()

    def get_v(node):
        if node.is_leaf:
            return node.value
        else:
            return node.threshold

    for d in range(depth):
        level = [Node(feature=0,
                      threshold=(get_v(level[2 * i]) + get_v(level[2 * i + 1])) / 2,
                      left_child=level[2 * i],
                      right_child=level[2 * i + 1], depth=depth - d - 1) for i in range(2 ** (depth - d - 1))]
    root = level[0]
    root.is_root = True
    return Decision_Tree(root=root)

def print_indicator_values_on_leaves(T,A):
    leaves=T.get_leaves()
    T.update_bounds()
    for leaf in leaves :
        leaf.update_indicator()
    print ("values of indicators of leaves :\n",np.array([leaf.indicator(A) for leaf in leaves]))

T=example_0()
A=np.array([[1,22000],[1,44000],[0,22000],[0,44000]])
print("\n\nFor example_0()")
print("A=\n",A)
print_indicator_values_on_leaves(T,A)

T=example_1(4)
A=np.array([[11.65],[6.917]])
print("\n\nFor example_1(4)")
print("A=\n",A)
print_indicator_values_on_leaves(T,A)
$ ./5-main.py


For example_0()
A=
 [[    1 22000]
 [    1 44000]
 [    0 22000]
 [    0 44000]]
values of indicators of leaves :
 [[ True  True False False]
 [False False False  True]
 [False False  True False]]


For example_1(4)
A=
 [[11.65 ]
 [ 6.917]]
values of indicators of leaves :
 [[False False]
 [False False]
 [False False]
 [ True False]
 [False False]
 [False False]
 [False False]
 [False False]
 [False  True]
 [False False]
 [False False]
 [False False]
 [False False]
 [False False]
 [False False]
 [False False]]
```

<b>Note: </b> there is exactly one `True` in each column (corresponding to the leaf the individual falls in).