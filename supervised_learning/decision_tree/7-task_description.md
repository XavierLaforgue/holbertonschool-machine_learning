Now we want to make our trees trainable, so we will write a method `Decision_Tree.fit` so that, when given  

- a 2D numpy array  `explanatory` of shape `(number of individuals, number of features)`.
- a 1D numpy array `target` of size `number of individuals`.

and evaluating the code below, should return a decision tree to make predictions.

```
T=Decision_Tree()
T.fit(explanatory,target)
```

### The `fit` function
The code below showcases the fit function. As you can observe, we assign a value to the attribute `self.root.sub_population`.  During the training, each node we build will have this attribute assigned with a 1D numpy array of booleans of size `target.size` (which is the number of individuals in the training set).
The `i`-th value of this array is `True` if and only if the `i`-th individual visits the node (so for the root, all the values are `True` as you can see). 

* To be added in the `Decision_Tree` class :
     
```
def fit(self,explanatory, target,verbose=0) :
		if self.split_criterion == "random" : 
				self.split_criterion = self.random_split_criterion
		else : 
				self.split_criterion = self.Gini_split_criterion     <--- to be defined later
		self.explanatory = explanatory
		self.target      = target
		self.root.sub_population = np.ones_like(self.target,dtype='bool')

		self.fit_node(self.root)     <--- to be defined later

		self.update_predict()     <--- defined in the previous task

		if verbose==1 :
				print(f"""  Training finished.
- Depth                     : { self.depth()       }
- Number of nodes           : { self.count_nodes() }
- Number of leaves          : { self.count_nodes(only_leaves=True) }
- Accuracy on training data : { self.accuracy(self.explanatory,self.target)    }""")     <--- to be defined later
```

### The `split` function

The training procedure consists in iteratively choosing splits from the root on, and the procedure to choose the splits depend on the situation, so, as you can see above, our training method will depend on an attribute `Decision_Tree.split_criterion`. For now, we will use a completely random way to split our nodes :

* To be added in the `Decision_Tree` class :
     
```
    def np_extrema(self,arr):
        return np.min(arr), np.max(arr)
				
    def random_split_criterion(self,node) :
        diff=0
        while diff==0 :
            feature=self.rng.integers(0,self.explanatory.shape[1])
            feature_min,feature_max=self.np_extrema(self.explanatory[:,feature][node.sub_population])
            diff=feature_max-feature_min
        x=self.rng.uniform()
        threshold= (1-x)*feature_min + x*feature_max
        return feature,threshold
```

<b> Note: </b>As surprising as it may be, and as we will check,  this randomized procedure already has an interesting predicting power.

### Task 

Finally, as you see, the fit method just initializes some attributes of the tree and then calls a new method `Decision_Tree.fit_node`  on the root. Your task is to update the class `Decision_Tree` by adding and completing the method `def fit_node(self,node) :`
 
- A node is a leaf if either it contains less than `min_pop` individuals, or its depth equals `max_depth` or all the individuals of the training set that come to this node are in the same class (i.e. have the same `target` value)
- The value to be computed for a leaf is the most represented class among the individuals that finish their trip in this leaf.
- At a node, the splitting criterion furnishes a feature index and a threshold. If the value of the selected feature on an individual that crosses this node is greater (strictly) than the threshold, then the individual goes in the left child, otherwise it goes in the right child.
-  No for loop on the individuals should appear in your code. Use numpy functions everywhere to get an efficient program.


```
def fit_node(self,node) :
		node.feature, node.threshold = self.split_criterion(node)

		left_population  =      <--- to be filled
		right_population =      <--- to be filled

		# Is left node a leaf ?
		is_left_leaf =    <--- to be filled

		if is_left_leaf :
				node.left_child = self.get_leaf_child(node,left_population)                                                         
		else :
				node.left_child = self.get_node_child(node,left_population)
				self.fit_node(node.left_child)

		# Is right node a leaf ?
		is_right_leaf =    <--- to be filled

		if is_right_leaf :
				node.right_child = self.get_leaf_child(node,right_population)
		else :
				node.right_child = self.get_node_child(node,right_population)
				self.fit_node(node.right_child)    

def get_leaf_child(self, node, sub_population) :        
		value =    <-- to be filled
		leaf_child= Leaf( value )
		leaf_child.depth=node.depth+1
		leaf_child.subpopulation=sub_population
		return leaf_child

def get_node_child(self, node, sub_population) :        
		n= Node()
		n.depth=node.depth+1
		n.sub_population=sub_population
		return n

def accuracy(self, test_explanatory , test_target) :
		return np.sum(np.equal(self.predict(test_explanatory), test_target))/test_target.size

```

### Main to test your work
<br>
**Main 1**

```
$ cat 7-main_1.py
#!/usr/bin/env python3

Decision_Tree = __import__('7-build_decision_tree').Decision_Tree
import numpy as np
from sklearn import datasets


#                                     #########################
#                                     # Generating examples : #
#                                     #########################

def circle_of_clouds(n_clouds, n_objects_by_cloud, radius=1, sigma=None, seed=0, angle=0):
    """
    This function returns a dataset made of 'n_clouds' classes.
    Each class is a small gaussian cloud containing 'n_objects_by_cloud' points.
    The centers of the clouds are regularly disposed on a circle of radius 'radius' (and center (0,0)).
    The spreadth of the clouds is governed by 'sigma'.
    """
    rng = np.random.default_rng(seed)
    if not sigma:
        sigma = np.sqrt(2 - 2 * np.cos(2 * np.pi / n_clouds)) / 7

    def rotate(x, k):
        theta = 2 * k * np.pi / n_clouds + angle
        m = np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        return np.matmul(x, m)

    def cloud():
        return (rng.normal(size=2 * n_objects_by_cloud) * sigma).reshape(n_objects_by_cloud, 2) + np.array([radius, 0])

    def target():
        return np.array(([[i] * n_objects_by_cloud for i in range(n_clouds)]), dtype="int32").ravel()

    return np.concatenate([np.array(rotate(cloud(), k)) for k in range(n_clouds)], axis=0), target()


def iris():
    """ Returns the explanatory features and the target of the famous iris dataset """
    iris = datasets.load_iris()
    return iris.data, iris.target


def wine():
    """ Returns the explanatory features and the target of the wine dataset """
    wine = datasets.load_wine()
    return wine.data, wine.target


#                                     #########################
#                                     #    Data preparation   #
#                                     #########################

def split(explanatory, target, seed=0, proportion=.1):
    """ Returns a dictionary containing a a training dataset and a test dataset """
    rng = np.random.default_rng(seed)
    test_indices = rng.choice(target.size, int(target.size * proportion), replace=False)
    test_filter = np.zeros_like(target, dtype="bool")
    test_filter[test_indices] = True

    return {"train_explanatory": explanatory[np.logical_not(test_filter), :],
            "train_target": target[np.logical_not(test_filter)],
            "test_explanatory": explanatory[test_filter, :],
            "test_target": target[test_filter]}

# Main 1

for d,name in zip([ split(*circle_of_clouds(10,30)) , split(*iris()), split(*wine()) ], ["circle of clouds", "iris dataset", "wine dataset"]) :
    print("-"*52+"\n"+name+" :")
    T=Decision_Tree(split_criterion="random",max_depth=20,seed=0)
    T.fit(d["train_explanatory"],d["train_target"],verbose=1)
    T.update_predict()
    print(f"    - Accuracy on test          : {T.accuracy(d['test_explanatory'],d['test_target'])}")
print("-"*52)
$ ./7-main_1.py
----------------------------------------------------
circle of clouds :
  Training finished.
    - Depth                     : 10
    - Number of nodes           : 81
    - Number of leaves          : 41
    - Accuracy on training data : 1.0
    - Accuracy on test          : 0.9666666666666667
----------------------------------------------------
iris dataset :
  Training finished.
    - Depth                     : 15
    - Number of nodes           : 43
    - Number of leaves          : 22
    - Accuracy on training data : 1.0
    - Accuracy on test          : 0.9333333333333333
----------------------------------------------------
wine dataset :
  Training finished.
    - Depth                     : 17
    - Number of nodes           : 137
    - Number of leaves          : 69
    - Accuracy on training data : 1.0
    - Accuracy on test          : 0.7058823529411765
----------------------------------------------------
```
<br>
**Main 2**

```
$ cat 7-main_2.py
#!/usr/bin/env python3

Decision_Tree = __import__('7-build_decision_tree').Decision_Tree
import numpy as np
import matplotlib.pyplot as plt

#                                     #########################
#                                     # Generating examples : #
#                                     #########################

def circle_of_clouds(n_clouds, n_objects_by_cloud, radius=1, sigma=None, seed=0, angle=0):
    """
    This function returns a dataset made of 'n_clouds' classes.
    Each class is a small gaussian cloud containing 'n_objects_by_cloud' points.
    The centers of the clouds are regularly disposed on a circle of radius 'radius' (and center (0,0)).
    The spreadth of the clouds is governed by 'sigma'.
    """
    rng = np.random.default_rng(seed)
    if not sigma:
        sigma = np.sqrt(2 - 2 * np.cos(2 * np.pi / n_clouds)) / 7

    def rotate(x, k):
        theta = 2 * k * np.pi / n_clouds + angle
        m = np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        return np.matmul(x, m)

    def cloud():
        return (rng.normal(size=2 * n_objects_by_cloud) * sigma).reshape(n_objects_by_cloud, 2) + np.array([radius, 0])

    def target():
        return np.array(([[i] * n_objects_by_cloud for i in range(n_clouds)]), dtype="int32").ravel()

    return np.concatenate([np.array(rotate(cloud(), k)) for k in range(n_clouds)], axis=0), target()

#                                     #########################
#                                     #    2D Visualization   #
#                                     #########################

def np_extrema(arr):
    return np.min(arr), np.max(arr)


def visualize_bassins(ax, model, x_min, x_max, y_min, y_max, cmap):
    """ color the points of a box
    with the color corresponding to the class predicted by the model """
    assert T.explanatory.shape[1] == 2, "Not a 2D example"
    X = np.linspace(x_min, x_max, 100)
    Y = np.linspace(y_min, y_max, 100)
    XX, YY = np.meshgrid(X, Y)
    XX_flat = XX.ravel()
    YY_flat = YY.ravel()
    Z = model.predict(np.vstack([XX_flat, YY_flat]).T)
    ax.pcolormesh(XX, YY, Z.reshape([100, 100]), cmap=cmap, shading='auto')


def visualize_training_dataset_2D(ax, model, cmap):
    """ color the points of the 'explanatory' array
    with the color corresponding to the class stored in 'target' """
    ax.scatter(model.explanatory[:, 0], model.explanatory[:, 1], c=model.target, cmap=cmap)


def visualize_model_2D(model, cmap=plt.cm.Set1):
    """ gather the results of visualize_bassins and visualize_training_dataset_2D """
    assert model.explanatory.shape[1] == 2, "Not a 2D example"

    x_min, x_max = np_extrema(model.explanatory[:, 0])
    y_min, y_max = np_extrema(model.explanatory[:, 1])
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    visualize_training_dataset_2D(axes[0], model, cmap)
    visualize_bassins(axes[1], model, x_min, x_max, y_min, y_max, cmap)
    plt.savefig("bassins2.png")
    plt.show()

#Main 2
explanatory,target = circle_of_clouds(10,30)
T=Decision_Tree(split_criterion="random")
T.fit(explanatory,target,verbose=0)
visualize_model_2D(T)
$ ./7-main_2.py
```

Main 2 should show the following plots. <br>
![bassins random.png](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2023/9/d9a40d4560ad056eba77d531a39e91c503d45a14.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20260818%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20260818T222344Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=2424dfbf3f99c5560d0c3b3d8aac6c44d7a86cfa1079d4a70ec8e3230b9c11bc)