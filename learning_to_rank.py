from pyspark.ml.base import Estimator
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import Evaluator

class LearningToRank(Estimator):
    """
    Class that implements different approaches of learning to rank algorithms.
    """

    def __init__(self, max_iter=10, reg_param=0.1, features_col="features", label_col="label"):
        """
        Init the learning-to-rank model.
        
        :param max_iter: the maximum number of iterations
        :param reg_param: regularization parameter
        :param features_col: the name of the column that contains the feature representation 
        the model will be trained on
        :param label_col: the name of the column that contains the class of each feature representation
        """
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.features_col = features_col
        self.label_col = label_col

    def _fit(self, dataset):
        lsvc = LinearSVC(maxIter=self.max_iter, regParam=self.reg_param, featuresCol=self.features_col, labelCol=self.label_col)
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
