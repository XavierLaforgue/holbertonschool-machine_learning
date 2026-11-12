
Now we are in good position to implement the class `Isolation_Forest` following the above directions. : 

* Complete the method  `def suspects(self,explanatory,n_suspects):`
* You should use the following imports:
	* `Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree`
	* `import numpy as np` 
   
```
class Isolation_Random_Forest() :
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0) :
        self.numpy_predicts  = []
        self.target          = None
        self.numpy_preds     = None
        self.n_trees         = n_trees
        self.max_depth       = max_depth
        self.seed            = seed
        
    def predict(self, explanatory):
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)
            
    def fit(self,explanatory,n_trees=100,verbose=0) :
        self.explanatory = explanatory
        self.numpy_preds = []
        depths           = [] 
        nodes            = [] 
        leaves           = []
        for i in range(n_trees) :
            T = Isolation_Random_Tree(max_depth=self.max_depth,seed=self.seed+i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(    T.depth()                         )
            nodes.append(     T.count_nodes()                   )
            leaves.append(    T.count_nodes(only_leaves=True)   )
        if verbose==1 :
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }""")
            
    def suspects(self,explanatory,n_suspects) :
				""" returns the n_suspects rows in explanatory that have the smallest mean depth """
        depths=self.predict(explanatory)
				pass          <--- to be filled
```
**Main to test your work**

```
$ cat 11-main.py
#!/usr/bin/env python3

Isolation_Random_Forest = __import__('11-isolation_forest').Isolation_Random_Forest

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

#Main
explanatory,_ = circle_of_clouds(3,100,sigma=.2)
IRF=Isolation_Random_Forest(max_depth=15)
IRF.fit(explanatory,verbose=1)
suspects,depths = IRF.suspects(explanatory,n_suspects=3)
print("suspects :",suspects)
print("depths of suspects :",depths)
#Visualization  
visualize_model_2D(IRF,cmap='RdBu')
$ ./11-main.py
  Training finished.
    - Mean depth                     : 15.0
    - Mean number of nodes           : 550.1
    - Mean number of leaves          : 275.55
suspects : [[ 0.09754323  1.33996024]
 [-0.95592937  1.23922096]
 [-0.36715428 -1.38766761]]
depths of suspects : [4.84 6.02 6.12]
```
This main should show the following plots. 
<br>
![bassins_isolation_Forest.png](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/medias/2023/9/d48cce1c9cce0c6597484b04f33a2f8e5d129b2c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20260818%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20260818T222344Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=b491ffb43824ec4fbb954170a6c7ae8ca56d436959da5c875df7fa3d3d3e00c5) <br>

<b>Warning:</b> Duplicates in dataset can cause the programs below to enter infinite loops. It is therefore important to check first that there are none.