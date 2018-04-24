from pyspark import keyword_only
from pyspark.ml.classification import JavaClassificationModel
from pyspark.ml.param.shared import *
from pyspark.ml.util import *
from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.common import inherit_doc

@inherit_doc
class TestLinearSVC(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasMaxIter,
                HasRegParam, HasTol, HasRawPredictionCol, HasFitIntercept, HasStandardization,
                HasWeightCol, HasAggregationDepth, HasThreshold, JavaMLWritable, JavaMLReadable):
    """
    .. note:: Experimental

    `Linear SVM Classifier <https://en.wikipedia.org/wiki/Support_vector_machine#Linear_SVM>`_

    This binary classifier optimizes the Hinge Loss using the OWLQN optimizer.
    Only supports L2 regularization currently.

    >>> from pyspark.sql import Row
    >>> from pyspark.ml.linalg import Vectors
    >>> df = sc.parallelize([
    ...     Row(label=1.0, features=Vectors.dense(1.0, 1.0, 1.0)),
    ...     Row(label=0.0, features=Vectors.dense(1.0, 2.0, 3.0))]).toDF()
    >>> svm = LinearSVC(maxIter=5, regParam=0.01)
    >>> model = svm.fit(df)
    >>> model.coefficients
    DenseVector([0.0, -0.2792, -0.1833])
    >>> model.intercept
    1.0206118982229047
    >>> model.numClasses
    2
    >>> model.numFeatures
    3
    >>> test0 = sc.parallelize([Row(features=Vectors.dense(-1.0, -1.0, -1.0))]).toDF()
    >>> result = model.transform(test0).head()
    >>> result.prediction
    1.0
    >>> result.rawPrediction
    DenseVector([-1.4831, 1.4831])
    >>> svm_path = temp_path + "/svm"
    >>> svm.save(svm_path)
    >>> svm2 = LinearSVC.load(svm_path)
    >>> svm2.getMaxIter()
    5
    >>> model_path = temp_path + "/svm_model"
    >>> model.save(model_path)
    >>> model2 = LinearSVCModel.load(model_path)
    >>> model.coefficients[0] == model2.coefficients[0]
    True
    >>> model.intercept == model2.intercept
    True

    .. versionadded:: 2.2.0
    """

    threshold = Param(Params._dummy(), "threshold",
                      "The threshold in binary classification applied to the linear model"
                      " prediction.  This threshold can be any real number, where Inf will make"
                      " all predictions 0.0 and -Inf will make all predictions 1.0.",
                      typeConverter=TypeConverters.toFloat)

    userIdCol = Param(Params._dummy(), "userIdCol",
                      "The threshold in binary classification applied to the linear model"
                      " prediction.  This threshold can be any real number, where Inf will make"
                      " all predictions 0.0 and -Inf will make all predictions 1.0.",
                      typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                 maxIter=100, regParam=0.0, tol=1e-6, rawPredictionCol="rawPrediction",
                 fitIntercept=True, standardization=True, threshold=0.0, weightCol=None,
                 aggregationDepth=2, userIdCol="user_id"):
        super(TestLinearSVC, self).__init__()
        print("Initiation of test linear SVM")
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.classification.LTRLinearSVC", self.uid)
        self._setDefault(maxIter=100, regParam=0.0, tol=1e-6, fitIntercept=True,
                         standardization=True, threshold=0.0, aggregationDepth=2,
                         userIdCol="user_id")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    @since("2.2.0")
    def setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                  maxIter=100, regParam=0.0, tol=1e-6, rawPredictionCol="rawPrediction",
                  fitIntercept=True, standardization=True, threshold=0.0, weightCol=None,
                  aggregationDepth=2):
        """
        setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction", \
                  maxIter=100, regParam=0.0, tol=1e-6, rawPredictionCol="rawPrediction", \
                  fitIntercept=True, standardization=True, threshold=0.0, weightCol=None, \
                  aggregationDepth=2):
        Sets params for Linear SVM Classifier.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        print("Creating model")
        return TestLinearSVCModel(java_model)


class TestLinearSVCModel(JavaModel, JavaClassificationModel, JavaMLWritable, JavaMLReadable):
    """
    .. note:: Experimental

    Model fitted by LinearSVC.

    .. versionadded:: 2.2.0
    """

    @property
    @since("2.2.0")
    def coefficients(self):
        """
        Model coefficients of Linear SVM Classifier.
        """
        return self._call_java("coefficients")

    @property
    @since("2.2.0")
    def intercept(self):
        """
        Model intercept of Linear SVM Classifier.
        """
        return self._call_java("intercept")