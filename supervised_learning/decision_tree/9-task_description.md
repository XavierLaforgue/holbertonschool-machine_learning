In this task, we will create a new class `Random_Forest`. 

When training an object of this class on a dataset, it will build a large list of decision trees with random splitting criterion. Then to predict the class of an individual, it will ask each of those trees its prediction, and will choose the prediction that is the most frequent.

**Pros** : this method has advantages over the use of the Gini criterion 
- when the training dataset is large : it can save CPU usage,
-  in terms of stability  : the result of this method should be almost the same on the various training subsets of a cross-validation procedure while the Gini based decision trees can be very different for each of these  training subsets.

**Cons** : The Gini-based decision tree furnishes a model that has a clear, elementary interpretation. This interpretation can be used, once the decision tree, to further understand (in a human sense) the dependence between the explanatory data and the target.

**Task:**  In the class `Random_Forest `  : <br>

*  Insert the below declarations <br>
*  Add the method `def predict(self, explanatory):` <br>
*  You should use the following import: <br>

	*  `Decision_Tree = __import__('8-build_decision_tree').Decision_Tree`
	* ` import numpy as np` <br>
	
```
class Random_Forest() :
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0) :
        self.numpy_predicts  = []
        self.target          = None
        self.numpy_preds     = None
        self.n_trees         = n_trees
        self.max_depth       = max_depth
        self.min_pop         = min_pop
        self.seed            = seed

    def predict(self, explanatory):            <--    to be filled
		
        # Initialize an empty list to store predictions from individual trees

        # Generate predictions for each tree in the forest
				
        # Calculate the mode (most frequent) prediction for each example
			  
    def fit(self,explanatory,target,n_trees=100,verbose=0) :
        self.target      = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths           = [] 
        nodes            = [] 
        leaves           = []
        accuracies =[]
        for i in range(n_trees) :
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop,seed=self.seed+i)
            T.fit(explanatory,target)
            self.numpy_preds.append(T.predict)
            depths.append(    T.depth()                         )
            nodes.append(     T.count_nodes()                   )
            leaves.append(    T.count_nodes(only_leaves=True)   )
            accuracies.append(T.accuracy(T.explanatory,T.target))
        if verbose==1 :
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }
    - Mean accuracy on training data : { np.array(accuracies).mean()  }
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,self.target)}""")
                               
    def accuracy(self, test_explanatory , test_target) :
        return np.sum(np.equal(self.predict(test_explanatory), test_target))/test_target.size
```
				
#### Main to test your work

<br>
**Main 1**

```
$ cat 9-main_1.py
#!/usr/bin/env python3

Random_Forest = __import__('9-random_forest').Random_Forest
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
    F=Random_Forest(max_depth=6)
    F.fit(d["train_explanatory"],d["train_target"],verbose=1)
    print(f"    - Accuracy on test          : {F.accuracy(d['test_explanatory'],d['test_target'])}")
print("-"*52)
$ ./9-main_1.py
----------------------------------------------------
circle of clouds :
  Training finished.
    - Mean depth                     : 6.0
    - Mean number of nodes           : 50.92
    - Mean number of leaves          : 25.96
    - Mean accuracy on training data : 0.8364814814814814
    - Accuracy of the forest on td   : 1.0
    - Accuracy on test          : 1.0
----------------------------------------------------
iris dataset :
  Training finished.
    - Mean depth                     : 6.0
    - Mean number of nodes           : 26.56
    - Mean number of leaves          : 13.78
    - Mean accuracy on training data : 0.884074074074074
    - Accuracy of the forest on td   : 0.9777777777777777
    - Accuracy on test          : 0.8666666666666667
----------------------------------------------------
wine dataset :
  Training finished.
    - Mean depth                     : 6.0
    - Mean number of nodes           : 37.08
    - Mean number of leaves          : 19.04
    - Mean accuracy on training data : 0.7626086956521739
    - Accuracy of the forest on td   : 1.0
    - Accuracy on test          : 0.9411764705882353
----------------------------------------------------
```

<br>
**Main 2**

```
$ cat 9-main_2.py
#!/usr/bin/env python3

Random_Forest = __import__('9-random_forest').Random_Forest
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


def visualize_bassins(ax, model, x_min, x_max, y_min, y_max,cmap) :
    """ color the points of a box
    with the color corresponding to the class predicted by the model """
    assert model.explanatory.shape[1]==2, "Not a 2D example"
    X       = np.linspace(x_min, x_max, 100)
    Y       = np.linspace(y_min, y_max, 100)
    XX,YY   = np.meshgrid(X, Y)
    XX_flat = XX.ravel()
    YY_flat = YY.ravel()
    Z       = model.predict(np.vstack([XX_flat, YY_flat]).T)
    ax.pcolormesh(XX, YY, Z.reshape([100,100]), cmap=cmap,shading='auto')


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
    
#Main2
explanatory,target = circle_of_clouds(10,30)
F=Random_Forest()
F.fit(explanatory,target,verbose=0)

visualize_model_2D(F)
$ ./9-main_2.py
```
Main 2 should show the following plots.<br>
![bassins forest.png](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2023/9/221439420292b9077bd31b5d70ab1c5e40f49df0.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20260818%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20260818T222344Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=433f28c2d8a72ffe9b5bc8398279ae21ebadbce9be57c7d722a43754dbf2fc19)

<b> NOTE: </b> Once again, we obtain very good results.