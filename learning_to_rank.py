from pyspark.ml.base import Estimator
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import Evaluator

class LearningToRank(Estimator):
    """
    Class that implements different approaches of learning to rank algorithms.
    """
    # TODO add comments
    def __init__(self, maxIter=10, regParam=0.1, featuresCol="features", labelCol="label"):
        self._maxIter = maxIter
        self._regParam = regParam
        self._featuresCol = featuresCol
        self._labelCol = labelCol

    def _fit(self, dataset):
        lsvc = LinearSVC(self.maxIter, self.regParam, self.featuresCol, self.labelCol)
        # Fit the model
        lsvcModel = lsvc.fit(dataset)
        return lsvcModel

    def setMaxIter(self, maxIter):
        self._maxIter = maxIter

    def maxIter(self):
        return self._maxIter

    def setRegParam(self, regParam):
        self._regParam = regParam

    def regParam(self):
        return self._regParam

    def setFeaturesCol(self, featuresCol):
        self._featuresCol = featuresCol

    def featuresCol(self):
        return self._featuresCol

    def setLabelCol(self, labelCol):
        self._labelCol = labelCol

    def labelCol(self):
        return self._labelCol

class LearningToRankEvaluator(Evaluator):
    """
    That that calculates how good are the predictions of learning-to-rank algorithm.
    """

    def _evaluate(self, dataset):
        """
        Evaluates the output.

        :param dataset: a dataset that contains labels/observations and
               predictions
        :return: metric
        """
        return 0;