We are now in a position to write our efficient `Decision_Tree.predict` function.

**Task:** Write a method `Decision_Tree.update_predict` that computes the prediction function :

* Fill in `def update_predict(self):` in class `Decision_Tree`:

```
def update_predict(self) :
        self.update_bounds()
        leaves=self.get_leaves()
        for leaf in leaves :
            leaf.update_indicator()          
        self.predict = lambda A: #<--- To be filled
```

<b> Main to test your work</b>

In this conept page: <a href="https://intranet.hbtn.io/concepts/1200"> Decision_Tree.pred vs Decision_Tree.predict
</a>, we introduced an additional approach for implementing a prediction function known as` Decision_Tree.pred` .<br>
As part of the testing process, insert the following methods into their respective classes:

*  add `def pred(self,x):` in class `Leaf`:

```
    def pred(self,x) :
        return self.value
```

* add `def pred(self,x):` in class `Node`:

```
    def pred(self,x) :
        if x[self.feature]>self.threshold :
            return self.left_child.pred(x)
        else :
            return self.right_child.pred(x)
```

* add `def pred(self,x):` in class `Decision_Tree`:

```
    def pred(self,x) :
            return self.root.pred(x)
```

Now, to validate whether `Decision_Tree.pred` performs similarly to the existing `Decision_Tree.predict`, we are creating a generator for random trees. We will compare the behavior of `Decision_Tree.predict` and `Decision_Tree.pred` on a sample explanatory array.

```
$ cat 6-main.py
#!/usr/bin/env python3

Node = __import__('6-build_decision_tree').Node
Leaf = __import__('6-build_decision_tree').Leaf
Decision_Tree = __import__('6-build_decision_tree').Decision_Tree
import numpy as np

def random_tree(max_depth, n_classes,n_features,seed=0) :
    assert max_depth>0, "max_depth must be a strictly positive integer"
    rng=np.random.default_rng(seed)
    root=Node(is_root=True,depth=0)
    root.lower={i:-100 for i in range(n_features)}
    root.upper={i:100 for i in range(n_features)}
    
    def build_children(node) :
        feat=rng.integers(0,n_features)
        node.feature=feat
        node.threshold = np.round(rng.uniform(0,1)*(node.upper[feat]-node.lower[feat])+node.lower[feat],2)
        if node.depth==max_depth-1 :
            node.left_child=Leaf(depth=max_depth, value=rng.integers(0,n_classes))
            node.right_child=Leaf(depth=max_depth, value=rng.integers(0,n_classes))
        else :
            node.left_child=Node(depth=node.depth+1)
            node.left_child.lower=node.lower.copy()
            node.left_child.upper=node.upper.copy()
            node.left_child.lower[feat]=node.threshold
            node.right_child=Node(depth=node.depth+1)
            node.right_child.lower=node.lower.copy()
            node.right_child.upper=node.upper.copy()
            node.right_child.upper[feat]=node.threshold
            build_children(node.left_child)
            build_children(node.right_child)
            
    T=Decision_Tree(root=root)
    build_children(root)
    
    A=rng.uniform(0,1,size=100*n_features).reshape([100,n_features])*200-100
    return T, A
    
T,A=random_tree(4, 3 ,5,seed=1)
print(T)

T.update_predict()

print("T.pred(A) :\n",np.array([T.pred(x) for x in A]))
print("T.predict(A) :\n",T.predict(A))

test=np.all(np.equal(T.predict(A),np.array([T.pred(x) for x in A])))                         
print(f"Predictions are the same on the explanatory array A : {test}")      
$ ./6-main.py
root [feature=2, threshold=90.09]
    +---> node [feature=2, threshold=91.52]
    |      +---> node [feature=4, threshold=-37.63]
    |      |      +---> node [feature=4, threshold=20.63]
    |      |      |      +---> leaf [value=0]
    |      |      |      +---> leaf [value=2]
    |      |      +---> node [feature=1, threshold=9.92]
    |      |             +---> leaf [value=1]
    |      |             +---> leaf [value=0]
    |      +---> node [feature=0, threshold=50.7]
    |             +---> node [feature=4, threshold=-34.05]
    |             |      +---> leaf [value=1]
    |             |      +---> leaf [value=1]
    |             +---> node [feature=3, threshold=-39.36]
    |                    +---> leaf [value=0]
    |                    +---> leaf [value=1]
    +---> node [feature=4, threshold=-19.38]
           +---> node [feature=0, threshold=-59.31]
           |      +---> node [feature=2, threshold=42.64]
           |      |      +---> leaf [value=0]
           |      |      +---> leaf [value=0]
           |      +---> node [feature=1, threshold=-2.96]
           |             +---> leaf [value=0]
           |             +---> leaf [value=2]
           +---> node [feature=3, threshold=44.96]
                  +---> node [feature=4, threshold=-56.37]
                  |      +---> leaf [value=2]
                  |      +---> leaf [value=0]
                  +---> node [feature=3, threshold=40.6]
                         +---> leaf [value=0]
                         +---> leaf [value=1]

T.pred(A) :
 [0 0 0 0 1 1 2 0 1 0 0 0 2 0 2 0 1 0 1 1 0 1 0 0 0 2 1 0 0 0 1 0 0 0 0 1 0
 1 0 1 0 1 1 0 1 2 0 1 1 0 0 1 0 1 0 1 0 1 1 1 0 0 0 2 0 0 0 1 0 2 2 0 0 0
 0 0 1 0 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 0 1 1 0 0 0 1]
T.predict(A) :
 [0 0 0 0 1 1 2 0 1 0 0 0 2 0 2 0 1 0 1 1 0 1 0 0 0 2 1 0 0 0 1 0 0 0 0 1 0
 1 0 1 0 1 1 0 1 2 0 1 1 0 0 1 0 1 0 1 0 1 1 1 0 0 0 2 0 0 0 1 0 2 2 0 0 0
 0 0 1 0 0 1 0 1 0 1 1 0 0 0 0 0 0 1 0 0 1 1 0 0 0 1]
Predictions are the same on the explanatory array A : True
```  

**NOTE:** Read the concept page <a href="https://intranet.hbtn.io/concepts/1200">  Decision_Tree.pred vs Decision_Tree.predict
</a> to understand how `Decision_Tree.predict` is better than `Decision_Tree.pred` when it comes to computational efficiency.
