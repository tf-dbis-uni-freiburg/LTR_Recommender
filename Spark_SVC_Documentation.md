# LinearSVC implementation in Spark

It optimizes the Hinge Loss using the OWLQN optimizer (Only supports L2 regularization)

- Init parameters:
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

- Method -> **train**(dataset: Dataset[_]): LinearSVCModel
Steps:

1. optimizer = new BreezeOWLQN (used OWLQN (Scalable training of L1-regularized log-linear models stat))
2. initialCoefWithIntercept - DenseVector with zeros for all iterations

```
val states = optimizer.iterations(new CachedDiffFunction(costFun),
        initialCoefWithIntercept.asBreeze.toDenseVector)
```

3.  For the best state that an optimizer found, divide the computed raw coefficient for a feature by its std. That's the coefficientVector
used by the model afterwards.

```
/*
The coefficients are trained in the scaled space; we're converting them back to
the original space.
Note that the intercept in scaled space and original space is the same;
as a result, no scaling is needed.
*/
val rawCoefficients = state.x.toArray
val coefficientArray = Array.tabulate(numFeatures) { i =>
    if (featuresStd(i) != 0.0) {
       rawCoefficients(i) / featuresStd(i)
    } else {
       0.0
    }
}
```

# RDD treeAggregate
Computes the same thing as aggregate, except it aggregates the elements of the RDD in a multi-level tree pattern.
Another difference is that it does not use the initial value for the second reduce function (combOp).
By default a tree of depth 2 is used, but this can be changed via the depth parameter.

- Method ->
    def **treeAggregate**[U](zeroValue: U)(seqOp: (U, T) ⇒ U, combOp: (U, U) ⇒ U, depth: Int = 2)(implicit arg0: ClassTag[U]): U

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

Used by OWLQN optimizer. Implements Breeze's DiffFunction[T] for hinge loss function.

- Init parameters:
1) **instances**: RDD
2) **fitIntercept**: Boolean
3) **standardization**: Boolean
4) **bcFeaturesStd**: Broadcast[Array[Double]] - broadcasted std over features of instances
5) **regParamL2**: Double
6) **aggregationDepth**: Int

- Method -> **calculate(coefficients:DenseVector[Double])** :return (Double, DenseVector[Double])

Steps:
1. broadcast input parameter coefficients (variable bcCoeffs)

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

           - update gradient for the coefficient by adding the multiplication of regParamL2 and the computed coefficient

```
totalGradientArray(index) += regParamL2 * value
```

           - add to the sum square of the coefficient

```
sum += value * value
```

       4.2.2. if no standardization - we still standardize the data to improve the rate of convergence; as a result, we have to perform this reverse standardization by penalizing each component
differently to get effectively the same objective function when the training dataset is not standardized.
        - if std for the feature (accessed based on the index of the coeff) == 0.0; add 0.0 to the sum; no update of gradient for this feature
        else
            - divide the coeff value by square of its std
            - update the coeff gradient by adding the multiplication of the computed dividion and the regParamL2
            - add to the sum the multiplicaion of coeff value and the computed division

```
val temp = value / (featuresStd(index) * featuresStd(index))
totalGradientArray(index) += regParamL2 * temp
value * temp
```

5. destroy bcCoeffs (only visible while calculate is executed)
6. return (svmAggregator.loss + regVal, new BDV(totalGradientArray)); return calculated loss + regulation value, and DenseVector of gradients

# LinearSVCAggregator class

It computes the gradient and loss for hinge loss function, as used in binary classification for instances in sparse or dense vector in an online fashion.
Two LinearSVCAggregator can be merged together to have a summary of loss and gradient of the corresponding joint dataset. This class standardizes
feature values during computation using bcFeaturesStd (broadcasted std array for each feature). Used by LinearSVCCostFun class.

- Init parameters:
1. **bcCoefficients**: Broadcast[Vector] broadcasted coefficients
2. **fitIntercept**: Boolean
4. **bcFeaturesStd**: Broadcast[Array[Double]] - broadcasted std over features of instances

- Private variables
1. weightSum: Double = 0.0
2. lossSum: Double = 0.0
3. coefficientsArray = bcCoefficients.value
4. gradientSumArray = new Array[Double](numFeaturesPlusIntercept)

- Method -> **add(instance: Instance):** :return this.type
> Method that is done over instance on one partition/executor. Add a new training instance to this LinearSVCAggregator, and update the loss and gradient of the objective function.

Steps:

    1. compute dotProduct variable - for each feature(if std for the featire is not 0.0 and the feature value is not 0.0) of the Instance add to the dotProduct
    ((broadcasted coefficient for this featire * value of the feature) / std of the feature)
```
if (localFeaturesStd(index) != 0.0 && value != 0.0) {
sum += localCoefficients(index) * value / localFeaturesStd(index)
}
```

    2. Our loss function with {0, 1} labels is max(0, 1 - (2y - 1) (f_w(x))). Therefore the gradient is -(2y - 1)*x. Compute loss using the calculated dotProduct (f_w(x)))


```
val labelScaled = 2 * label - 1.0
val loss = if (1.0 > labelScaled * dotProduct) {
  weight * (1.0 - labelScaled * dotProduct)
} else {
  0.0
 }}
```

    3. If there is a loss form this instance, update the gradient for this instance; gradientScale = -(2y - 1); For each feature, update the add to the gradient sum for this feature
    ((value of the feature) * gradientScale)/ std for this feature

```
if (1.0 > labelScaled * dotProduct) {
    val gradientScale = -labelScaled * weight
    features.foreachActive { (index, value) =>
     if (localFeaturesStd(index) != 0.0 && value != 0.0) {
        localGradientSumArray(index) += value * gradientScale / localFeaturesStd(index)
      }
    }
    if (fitIntercept) {
      localGradientSumArray(localGradientSumArray.length - 1) += gradientScale
    }
}
```

    4. update the overall loss by adding of loss for this instance
    5. update the overall weightSum by adding weight of the instance. In our case each instance has weight one. So weightSum for a LinearSVCAggregator is equal to the number of its instances.

> Method that is done for merging results for two partitions/executor. Merge another LinearSVCAggregator, and update the loss and gradient of the objective function.
    (Note that it's in place merging; as a result, `this` object will be modified.

- Method -> **merge(other: LinearSVCAggregator)** :return this.type
    1. Update the weightSum by summing with the weightSum of the other LinearSVCAggregator
    2. Update the lossSum by summing with the lossSum of the other LinearSVCAggregator
    3. Update gradientSumArray by adding other.gradientSumArray
```
if (other.weightSum != 0.0) {
   weightSum += other.weightSum
   lossSum += other.lossSum

   var i = 0
   val localThisGradientSumArray = this.gradientSumArray
   val localOtherGradientSumArray = other.gradientSumArray
   val len = localThisGradientSumArray.length
   while (i < len) {
    localThisGradientSumArray(i) += localOtherGradientSumArray(i)
    i += 1
   }
}
```
- Method -> **loss()** :return Double
```
if (weightSum != 0) lossSum / weightSum else 0.0
```

- Method -> **gradient()** :return Vector

1. if something was calculated in LinearSVCAggregator (weightSum != 0), return gradientVector computed in the aggregator scaled by 1/#number of instances calculated in the aggregator

```
val result = Vectors.dense(gradientSumArray.clone())
scal(1.0 / weightSum, result)]
```

2. Otherwise: Empty DenseVector

# OWLQN class (Scalable training of L1-regularized log-linear models stat)
TODO - read the paper and add notes for the optimizer