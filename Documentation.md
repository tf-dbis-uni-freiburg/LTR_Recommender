# LinearSVC implementation in Spark

- it optimizes the Hinge Loss using the OWLQN optimizer (Only supports L2 regularization)

- Parameters:
1) **featuresCol** - features column name. Default: ”features”
2) **labelCol** - label column name. Default: ”label”
3) **predictionCol** - prediction column name. Default: ”prediction”
4) **maxIter** - max number of iterations (>= 0). Default: 100
5) **regParam** - regularization parameter (>= 0). Default: 0.0
6) **tol** - the convergence tolerance for iterative algorithms (>= 0). Smaller values will lead to higher accuracy at the cost of more iterations.
Default: 1e-6
7) **rawPredictionCol** - raw prediction (a.k.a. confidence) column name. Default: ”rawPrediction”
8) **fitIntercept** - whether to fit an intercept term. Default: True
9) **standardization** - whether to standardize the training features before fitting the model. Default:True
10) **threshold** - The threshold in binary classification applied to the linear model prediction. This threshold can be any real number,
where Inf will make all predictions 0.0 and -Inf will make all predictions 1.0. If rawPrediction[1] > threshold, prediction 1; otherwise prediction 0.
Default value: 0.0
11) **weightCol** - weight column name. If this is not set or empty, we treat all instance weights as 1.0. Default: None
12) **aggregationDepth**- suggested depth for treeAggregate (>= 2). If the dimensions of features or the number of partitions are large,
this param could be adjusted to a larger size. Default: 2.
    > ! TreeReduce and TreeAggregate - In a regular reduce or aggregate functions in Spark (and the original MapReduce) all partitions have to send
    their reduced value to the driver machine, and that machine spends linear time on the number of partitions (due to the CPU cost in merging partial results
    and the network bandwidth limit). It becomes a bottleneck when there are many partitions and the data from each partition is big. Since Spark 1.1 was
    introduced a new aggregation communication pattern based on multi-level aggregation trees. In this setup, data are combined partially on a small set of
    executors before they are sent to the driver, which dramatically reduces the load the driver has to deal with. Tests showed that these functions reduce
    the aggregation time by an order of magnitude, especially on datasets with a large number of partitions.
    So, in treeReduce and in treeAggregate, the partitions talk to each other in a logarithmic number of rounds.

# RDD treeAggregate
Computes the same thing as aggregate, except it aggregates the elements of the RDD in a multi-level tree pattern.
Another difference is that it does not use the initial value for the second reduce function (combOp).
By default a tree of depth 2 is used, but this can be changed via the depth parameter.

- Listing Variants:
    def treeAggregate[U](zeroValue: U)(seqOp: (U, T) ⇒ U, combOp: (U, U) ⇒ U, depth: Int = 2)(implicit arg0: ClassTag[U]): U

- Example:
```scala
val z = sc.parallelize(List(1,2,3,4,5,6), 2)

// lets first print out the contents of the RDD with partition labels
def myfunc(index: Int, iter: Iterator[(Int)]) : Iterator[String] = {
   iter.map(x => "[partID:" +  index + ", val: " + x + "]")
}

z.mapPartitionsWithIndex(myfunc).collect
res28: Array[String] = Array([partID:0, val: 1], [partID:0, val: 2], [partID:0, val: 3], [partID:1, val: 4], [partID:1, val: 5], [partID:1, val: 6])

z.treeAggregate(0)(math.max(_, _), _ + _)
res40: Int = 9

// Note unlike normal aggregrate. Tree aggregate does not apply the initial value for the second reduce
// This example returns 11 since the initial value is 5
// reduce of partition 0 will be max(5, 1, 2, 3) = 5
// reduce of partition 1 will be max(4, 5, 6) = 6
// final reduce across partitions will be 5 + 6 = 11
// note the final reduce does not include the initial value
z.treeAggregate(5)(math.max(_, _), _ + _)
res42: Int = 11
```
# LinearSVCCostFun class

- implements Breeze's DiffFunction[T] for hinge loss function
- parameters:
1) **instances**: RDD
2) **fitIntercept**: Boolean
3) **standardization**: Boolean
4) **bcFeaturesStd**: Broadcast[Array[Double]] - broadcasted std over features of instances
5) **regParamL2**: Double
6) **aggregationDepth**: Int

- **calculate(coefficients:DenseVector[Double])** :return (Double, DenseVector[Double])
Steps:
1. !!! broadcast input parameter coefficients (variable bcCoeffs)

2. create svmAggregator - using LinearSVCAggregator
- on each partition -> (c: LinearSVCAggregator, instance: Instance) => c.add(instance)
- combine step between partitions -> (c1: LinearSVCAggregator, c2: LinearSVCAggregator) => c1.merge(c2)
- zero value -> new LinearSVCAggregator(bcCoeffs, bcFeaturesStd, fitIntercept)

3. take -> totalGradientArray = svmAggregator.gradient.toArray; gradient array computed by svmAggregator
4. calculate regVal - regVal is the sum of coefficients squares excluding intercept for L2 regularization

 4.1. regVal = 0.0 if(regParamL2 == 0.0) else: 0.5 * regParamL2 * sum

 4.2. calculation of sum variable

for each coefficients (index, value) (not from the broadcasted, from the input coeffs)

4.2.1. if standardization

4.2.1.1. update gradient for the coefficient by adding the multiplication of regParamL2 and the computed coefficient

```
totalGradientArray(index) += regParamL2 * value
```

4.2.1.2. add to the sum square of the coefficient

```
sum += value * value
```

4.2.2. if no standardization - we still standardize the data to improve the rate of convergence; as a result, we have to perform this reverse standardization by penalizing each component
differently to get effectively the same objective function when the training dataset is not standardized.

4.2.2.1. if std for the feature (accessed based on the index of the coeff) == 0.0; add 0.0 to the sum; no update of gradient for this feature
else

4.2.2.2.
- divide the coeff value by square of its std
- update the coeff gradient by adding the multiplication of the computed dividion and the regParamL2
- add to the sum the multiplicaion of coeff value and the computed division

```
val temp = value / (featuresStd(index) * featuresStd(index))
totalGradientArray(index) += regParamL2 * temp
value * temp
```

5. !!! destroy bcCoeffs (only visible while calculate is executed)
6. return (svmAggregator.loss + regVal, new BDV(totalGradientArray)); return calculated loss + regulation value, and DenseVector of gradients

# LinearSVCAggregator class

It computes the gradient and loss for hinge loss function, as used in binary classification for instances in sparse or dense vector in an online fashion.
Two LinearSVCAggregator can be merged together to have a summary of loss and gradient of the corresponding joint dataset. This class standardizes
feature values during computation using bcFeaturesStd.
It has
# optimizer - OWLQN (Scalable training of L1-regularized log-linear models stat)