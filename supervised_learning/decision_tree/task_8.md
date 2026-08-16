8. Using Gini impurity function as a splitting criterion

For a node \(N\) containing a population \(P\) that is partitioned in \(k+1\) classes : \(P=P_0\sqcup P_1\sqcup\cdots\sqcup P_k\) , the Gini impurity of \(N\) is defined as

$$\text{Gini}(N)=1 - \left(\frac {\text{card}(P_0)} {\text{card}(P)} \right)^2 - \cdots - \left(\frac {\text{card}(P_k)} {\text{card}(P)} \right)^2 $$

The idea behind this definition is that

if the population of a node is equally partitioned into many classes, the Gini impurity will be large
if the population of a node comes mainly from one class, the Gini impurity will be small
So

if the Gini impurity of a leaf is large, we cannot be very confident in the prediction function of this node
if the Gini impurity of a leaf is small, we can have more confidence in the prediction function of this node
Hence the idea to split a node is to choose the feature and the threshold for which the average of the Gini impurities of the corresponding children is the smallest.

$$\text{Gini\_split}(N) = \frac{\text{card(left\_child)}}{\text{card}(P)} \text{Gini}(\text{left\_child}) + \frac {\text{card(right\_child)}} {\text{card}(P)} \text{Gini}(\text{right\_child})$$

Task: To find this value :

Update the the Decision_Tree class by adding the new methods down below.
Fill in the gap in the method def Gini_split_criterion_one_feature(self,node,feature) :.
No for or while loop allowed !
