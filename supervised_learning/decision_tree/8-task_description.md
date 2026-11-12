<head>
			<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
			<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>

<p>For a node <span class="math display">\(N\)</span> containing a population <span class="math display">\(P\)</span>  that is partitioned in <span class="math display">\(k+1\)</span>  classes : <span class="math display">\(P=P_0\sqcup P_1\sqcup\cdots\sqcup P_k\)</span> , the Gini impurity of <span class="math display">\(N\)</span> is defined  as</p>


<p><span class="math display"> $$\text{Gini}(N)=1 - \left(\frac {\text{card}(P_0)} {\text{card}(P)} \right)^2 - \cdots -  \left(\frac {\text{card}(P_k)} {\text{card}(P)} \right)^2 $$ </span></p>

The idea behind this definition is that 

 - if the population of a node is equally partitioned into many classes, the Gini impurity will be large
 - if the population of a node comes mainly from one class, the Gini impurity will be small
 
So 

 - if the Gini impurity of a leaf is large, we cannot be very confident in the prediction function of this node 
 - if the Gini impurity of a leaf is small, we can have more confidence in the prediction function of this node 

Hence the idea to split a node is to choose the feature and the threshold for which the average of the Gini impurities of the corresponding children is the smallest. 

<p><span class="math display"> $$\text{Gini_split}(N) = \frac {\text{card(left_child)}} {\text{card}(P)} \text{Gini}(\text{left_child}) + \frac {\text{card(right_child)}} {\text{card}(P)} \text{Gini}(\text{right_child}) $$</span></p>



<b>Task: </b> To find this value : 

* Update the the `Decision_Tree` class by adding the new methods down below.
* Fill in the gap in the method  `def Gini_split_criterion_one_feature(self,node,feature) :`. 
* No for or while loop allowed !

```
def possible_thresholds(self,node,feature) :
		values = np.unique((self.explanatory[:,feature])[node.sub_population])
		return (values[1:]+values[:-1])/2

def Gini_split_criterion_one_feature(self,node,feature) :
		# Compute a numpy array of booleans Left_F of shape (n,t,c) where
		#    -> n is the number of individuals in the sub_population corresponding to node
		#    -> t is the number of possible thresholds
		#    -> c is the number of classes represented in node
		# such that Left_F[ i , j , k] is true iff 
		#    -> the i-th individual in node is of class k 
		#    -> the value of the chosen feature on the i-th individual 
		#                              is greater than the t-th possible threshold
		# then by squaring and summing along 2 of the axes of Left_F[ i , j , k], 
		#                     you can get the Gini impurities of the putative left childs
		#                    as a 1D numpy array of size t 
		#
		# Then do the same with the right child
		# Then compute the average sum of these Gini impurities
		#
		# Then  return the threshold and the Gini average  for which the Gini average is the smallest

def Gini_split_criterion(self,node) :
		X=np.array([self.Gini_split_criterion_one_feature(node,i) for i in range(self.explanatory.shape[1])])
		i =np.argmin(X[:,1])
		return i, X[i,0]
```

#### Main to test your work
<br>
**Main 1**

```
$ cat 8-main_1.py
#!/usr/bin/env python3

Decision_Tree = __import__('8-build_decision_tree').Decision_Tree
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
    T=Decision_Tree(split_criterion="Gini",max_depth=20,seed=0)
    T.fit(d["train_explanatory"],d["train_target"],verbose=1)
    T.update_predict()
    print(f"    - Accuracy on test          : {T.accuracy(d['test_explanatory'],d['test_target'])}")
print("-"*52)
$ ./8-main_1.py
----------------------------------------------------
circle of clouds :
  Training finished.
    - Depth                     : 5
    - Number of nodes           : 19
    - Number of leaves          : 10
    - Accuracy on training data : 1.0
    - Accuracy on test          : 1.0
----------------------------------------------------
iris dataset :
  Training finished.
    - Depth                     : 5
    - Number of nodes           : 13
    - Number of leaves          : 7
    - Accuracy on training data : 1.0
    - Accuracy on test          : 0.9333333333333333
----------------------------------------------------
wine dataset :
  Training finished.
    - Depth                     : 5
    - Number of nodes           : 21
    - Number of leaves          : 11
    - Accuracy on training data : 1.0
    - Accuracy on test          : 0.9411764705882353
----------------------------------------------------
```
<br>
**Main 2**

```
$ cat 8-main_2.py
#!/usr/bin/env python3

Decision_Tree = __import__('8-build_decision_tree').Decision_Tree
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
T=Decision_Tree(split_criterion="Gini")
T.fit(explanatory,target,verbose=0)

visualize_model_2D(T)
$ ./8-main_2.py
```
Main 2 should show the following plots. <br>
![bassins Gini.png](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2023/9/2711c285fd42e946b9a7ef315483a3f55c5b5b2c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20260818%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20260818T222344Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=ebd0fd1ceafa2b2a767bca576980b441289e07d7662ce8d13b7f7351153a8c15)

<b>NOTE:</b> We observe that the decision trees constructed with the `Gini_split_criterion` are less prone to overfitting and have a smaller depth than the ones obtained with the  `random_split_criterion`.