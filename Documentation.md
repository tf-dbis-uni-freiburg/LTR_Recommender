# LinearSVC implementation in Spark

- param threshold - this threshold is applied to the rawPrediction, default is 0.0
- it optimizes the Hinge Loss using the OWLQN optimizer (Only supports L2 regularization)
- Parameters:
1) featuresCol - features column name. Default: ”features”
2) labelCol - label column name. Default: ”label”
3) predictionCol - prediction column name. Default: ”prediction”
4) maxIter - max number of iterations (>= 0). Default: 100
5) regParam - regularization parameter (>= 0). Default: 0.0
6) tol - the convergence tolerance for iterative algorithms (>= 0). Smaller values will lead to higher accuracy at the cost of more iterations.
Default: 1e-6
7) rawPredictionCol - raw prediction (a.k.a. confidence) column name. Default: ”rawPrediction”
8) fitIntercept - whether to fit an intercept term. Default: True
9) standardization - whether to standardize the training features before fitting the model. Default:True
10) threshold - The threshold in binary classification applied to the linear model prediction. This threshold can be any real number,
where Inf will make all predictions 0.0 and -Inf will make all predictions 1.0. If rawPrediction[1] > threshold, prediction 1; otherwise prediction 0.
Default value: 0.0
11) weightCol - weight column name. If this is not set or empty, we treat all instance weights as 1.0. Default: None
12) aggregationDepth- suggested depth for treeAggregate (>= 2). If the dimensions of features or the number of partitions are large,
this param could be adjusted to a larger size. Default: 2.
    ! TreeReduce and TreeAggregate - In a regular reduce or aggregate functions in Spark (and the original MapReduce) all partitions have to send
    their reduced value to the driver machine, and that machine spends linear time on the number of partitions (due to the CPU cost in merging partial results
    and the network bandwidth limit). It becomes a bottleneck when there are many partitions and the data from each partition is big. Since Spark 1.1 was
    introduced a new aggregation communication pattern based on multi-level aggregation trees. In this setup, data are combined partially on a small set of
    executors before they are sent to the driver, which dramatically reduces the load the driver has to deal with. Tests showed that these functions reduce
    the aggregation time by an order of magnitude, especially on datasets with a large number of partitions.
    So, in treeReduce and in treeAggregate, the partitions talk to each other in a logarithmic number of rounds.

# LinearSVCAggregator computes the gradient and loss for hinge loss function, as used in binary classification for instances in sparse or dense vector in an online fashion.
Two LinearSVCAggregator can be merged together to have a summary of loss and gradient of the corresponding joint dataset. This class standardizes
feature values during computation using bcFeaturesStd.
# optimizer - OWLQN (Scalable training of L1-regularized log-linear models stat)