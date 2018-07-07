from py4j.java_gateway import JavaObject

from pyspark import RDD, SparkContext, PickleSerializer
from pyspark.mllib.common import _py2java, _java2py
from pyspark.mllib.linalg import  _convert_to_vector
from pyspark.serializers import AutoBatchedSerializer
from pyspark.sql import DataFrame
import sys

from logger import Logger

if sys.version >= '3':
    long = int
    unicode = str

class UserLabeledPoint(object):

    """
    Class that represents the features and labels of a data point. Moreover, it takes into account
    a user id to which each data point belongs to.

    :param userId: 
    identifier of an user to which a data point belongs
    :param label:
      Label for this data point.
    :param features:
      Vector of features for this point (NumPy array, list,
      pyspark.mllib.linalg.SparseVector, or scipy.sparse column matrix).
    """

    def __init__(self, userId, label, features):
        self.userId = int(userId)
        self.label = float(label)
        self.features = _convert_to_vector(features)

    def __reduce__(self):
        return (UserLabeledPoint, (self.userId, self.label, self.features))

    def __str__(self):
        return "(" + ",".join((str(self.userId), str(self.label), str(self.features))) + ")"

    def __repr__(self):
        return "UserLabeledPoint(%s, %s, %s)" % (self.userId, self.label, self.features)


class LTRSVMModel():
    """
       A linear model that has a map of coefficients and a map of intercepts.
       For each user in the trained data set, there is an entry in both maps.
       These entries contain model information for a particular user. If a prediction
       needs to be done, it depends on the user id which a paper is connected to.
    """

    def __init__(self, weights, intercept):
        self.modelWeights = {}
        self.modelIntercepts = {}
        # weights
        keys = str(weights.keys())[4:-1].split(", ")
        for key in keys:
            vector = str(weights.get(int(key)))[6:-2].split(",")
            self.modelWeights[int(key)] = _convert_to_vector(vector)
        # intercepts
        interceptsKeys = str(intercept.keys())[4:-1].split(", ")
        for key in interceptsKeys:
            self.modelIntercepts[int(key)] = float(str(intercept.get(int(key)))[5:-1])
        self.threshold = None


    # x has to be (userId, vector)
    def predict(self, x):
        """
        Predict values for an RDD of points
        using the models trained.
        """
        if isinstance(x, RDD):
            return x.map(lambda userId, v: self.predict(userId, v))

    def predict(self, userId, x):
        """
        Predict a value for a single point using the user's model trained.
        """
        x = _convert_to_vector(x)
        margin = self.modelWeights[int(userId)].dot(x) # + self.modelIntercepts[int(userId)]
        if self.threshold is None:
            return margin
        else:
            return 1 if margin > self.threshold else 0

class LTRSVMWithSGD(object):

    def train(cls, data, iterations=100, step=1.0, regParam=0.01,
              miniBatchFraction=1.0, regType="l2", intercept=False, validateData=False, convergenceTol=0.001):
        """
        Train a support vector machine on the given data.

        :param data:
          The training data, an RDD of UserLabeledPoint.
        :param iterations:
          The number of iterations.
          (default: 100)
        :param step:
          The step parameter used in SGD.
          (default: 1.0)
        :param regParam:
          The regularizer parameter.
          (default: 0.01)
        :param miniBatchFraction:
          Fraction of data to be used for each SGD iteration.
          (default: 1.0)
        :param initialWeights:
          The initial weights.
          (default: None)
        :param regType:
          The type of regularizer used for training our model.
          Allowed values:

            - "l1" for using L1 regularization
            - "l2" for using L2 regularization (default)
            - None for no regularization
        :param intercept:
          Boolean parameter which indicates the use or not of the
          augmented representation for training data (i.e. whether bias
          features are activated or not).
          (default: False)
        :param validateData:
          Boolean parameter which indicates if the algorithm should
          validate data before training.
          (default: True)
        :param convergenceTol:
          A condition which decides iteration termination.
          (default: 0.001)
        """
        def train(rdd):
            return testCallMLlibFunc("trainLTRSVMModelWithSGD", rdd, int(iterations), float(step),
                                 float(regParam), float(miniBatchFraction), regType,
                                 bool(intercept), bool(validateData), float(convergenceTol))

        return _svm_regression_train_wrapper(train, LTRSVMModel, data)

# train_func should take two parameters, namely data and initial_weights, and
# return the result of a call to the appropriate JVM stub.
# _regression_train_wrapper is responsible for setup and error checking.
def _svm_regression_train_wrapper(train_func, modelClass, data):
    first = data.first()
    if not isinstance(first, UserLabeledPoint):
       raise TypeError("data should be an RDD of UserLabeledPoint, but got %s" % type(first))
    weights, intercept = train_func(data)
    return modelClass(weights, intercept)

def testCallMLlibFunc(name, *args):
    """ Call API in PythonMLLibAPI """
    sc = SparkContext.getOrCreate()
    api = getattr(sc._jvm.LTRPythonMLLibAPI(), name)
    return svmCallJavaFunc(sc, api, *args)

def svmCallJavaFunc(sc, func, *args):
    """ Call Java Function """
    args = [_svm_py2java(sc, a) for a in args]
    return _java2py(sc, func(*args))

def _svm_py2java(sc, obj):
    """ Convert Python object into Java """
    if isinstance(obj, RDD):
        obj = _svm_to_java_object_rdd(obj)
    elif isinstance(obj, DataFrame):
        obj = obj._jdf
    elif isinstance(obj, SparkContext):
        obj = obj._jsc
    elif isinstance(obj, list):
        obj = [_py2java(sc, x) for x in obj]
    elif isinstance(obj, JavaObject):
        pass
    elif isinstance(obj, (int, long, float, bool, bytes, unicode)):
        pass
    else:
        data = bytearray(PickleSerializer().dumps(obj))
        obj = sc._jvm.org.apache.spark.mllib.api.python.LTRSerDe.loads(data)
    return obj

# this will call the MLlib version of pythonToJava()
def _svm_to_java_object_rdd(rdd):
    """ Return a JavaRDD of Object by unpickling

    It will convert each Python object into Java object by Pyrolite, whenever the
    RDD is serialized in batch or not.
    """
    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))
    return rdd.ctx._jvm.org.apache.spark.mllib.api.python.LTRSerDe.pythonToJava(rdd._jrdd, True)