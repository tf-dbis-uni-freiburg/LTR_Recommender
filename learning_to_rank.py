from pyspark.ml.base import Transformer
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import Evaluator

"""
Class that implements different approaches of learning to rank algorithms.
"""
class LearningToRank(Transformer):

    def _fit(self, dataset):
        #TODO add possibility to configute featureCol dynamically
        lsvc = LinearSVC(maxIter=10, regParam=0.1, featuresCol="paper_pair_diff", labelCol="label")
        # Fit the model
        lsvcModel = lsvc.fit(dataset)
        return lsvcModel

"""
That that calculates how good are the predictions of learning-to-rank algorithm.
"""
class LearningToRankEvaluator(Evaluator):

    def _evaluate(self, dataset):
        """
        Evaluates the output.

        :param dataset: a dataset that contains labels/observations and
               predictions
        :return: metric
        """
        return 0;