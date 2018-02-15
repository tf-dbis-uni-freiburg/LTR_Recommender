from pyspark.ml.base import Estimator
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import Evaluator

class LearningToRank(Estimator):
    """
    Class that implements different approaches of learning to rank algorithms.
    """
    def _fit(self, dataset):
        #TODO add possibility to configute featureCol dynamically
        lsvc = LinearSVC(maxIter=10, regParam=0.1, featuresCol="paper_pair_diff", labelCol="label")
        # Fit the model
        lsvcModel = lsvc.fit(dataset)
        return lsvcModel

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