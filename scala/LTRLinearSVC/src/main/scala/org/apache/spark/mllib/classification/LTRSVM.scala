package org.apache.spark.mllib.classification
import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.classification.impl.GLMClassificationModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.{DataValidators, Loader, Saveable}
import org.apache.spark.rdd.RDD

/**
 * Model for Support Vector Machines (SVMs).
 *
 * @param weights map that contains user id and its weights computed for every feature.
 * @param intercept map that contains user id and its intercept computed for this model.
 */
class LTRSVMModel (
   override val weights: Map[Int, Vector],
   override val intercept: Map[Int, Double])
  extends LTRGeneralizedLinearModel(weights, intercept) with Serializable with PMMLExportable {

  private var threshold: Option[Double] = Some(0.0)

  /**
   * Sets the threshold that separates positive predictions from negative predictions. An example
   * with prediction score greater than or equal to this threshold is identified as a positive,
   * and negative otherwise. The default value is 0.0.
   */
  def setThreshold(threshold: Double): this.type = {
    this.threshold = Some(threshold)
    this
  }

  /**
   * Returns the threshold (if any) used for converting raw prediction scores into 0/1 predictions.
   */

  def getThreshold: Option[Double] = threshold

  /**
   * Clears the threshold so that `predict` will output raw prediction scores.
   */
  def clearThreshold(): this.type = {
    threshold = None
    this
  }

  override protected def predictPoint(userId: Int, dataMatrix: Vector, weightMatrix: Map[Int, Vector], intercept: Map[Int, Double]) = {
    val weightVector = weightMatrix.get(userId).get
    val margin = weightVector.asBreeze.dot(dataMatrix.asBreeze) + intercept.get(userId).get
    threshold match {
      case Some(t) => if (margin > t) 1.0 else 0.0
      case None => margin
    }
  }

  override def toString: String = {
    s"${super.toString}, numClasses = 2, threshold = ${threshold.getOrElse("None")}"
  }
}

/**
 * Train a Support Vector Machine (SVM) using Stochastic Gradient Descent. By default L2
 * regularization is used, which can be changed via `LTRSVMWithSGD.optimizer`.
 *
 * @note Labels used in SVM should be {0, 1}.
 */
class LTRSVMWithSGD private (
    private var stepSize: Double,
    private var numIterations: Int,
    private var regParam: Double,
    private var miniBatchFraction: Double)
  extends LTRGeneralizedLinearAlgorithm[LTRSVMModel] with Serializable {

  private val gradient = new HingeGradient()
  private val updater = new SquaredL2Updater()
  
  override val optimizer = new LTRGradientDescent(gradient, updater)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setRegParam(regParam)
    .setMiniBatchFraction(miniBatchFraction)
    
  /**
   * Construct a LTRSVM object with default parameters: {stepSize: 1.0, numIterations: 100,
   * regParm: 0.01, miniBatchFraction: 1.0}.
   */
  def this() = this(1.0, 100, 0.01, 1.0)

  override protected def createModel(weights: Map[Int, Vector], intercept: Map[Int, Double]) = {
    new LTRSVMModel(weights, intercept)
  }
}

/**
 * Top-level methods for calling LTRSVM.
 *
 * @note Labels used in SVM should be {0, 1}.
 */
object LTRSVMWithSGD {

  /**
   * Train a SVM model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient. The weights used in
   * gradient descent are initialized using the initial weights provided.
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param regParam Regularisation parameter.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   * @param initialWeights Initial set of weights to be used. Array should be equal in size to
   *        the number of features in the data.
   *
   * @note Labels used in SVM should be {0, 1}.
   */
  def train(
      input: RDD[UserLabeledPoint],
      numIterations: Int,
      stepSize: Double,
      regParam: Double,
      miniBatchFraction: Double,
      initialWeights: Map[Int, Vector]): LTRSVMModel = {
    new LTRSVMWithSGD(stepSize, numIterations, regParam, miniBatchFraction)
      .run(input, initialWeights)
  }

  /**
   * Train a SVM model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate the gradient.
   *
   * @note Labels used in SVM should be {0, 1}
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param regParam Regularization parameter.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   */
  def train(
      input: RDD[UserLabeledPoint],
      numIterations: Int,
      stepSize: Double,
      regParam: Double,
      miniBatchFraction: Double): LTRSVMModel = {
    new LTRSVMWithSGD(stepSize, numIterations, regParam, miniBatchFraction).run(input)
  }

  /**
   * Train a SVM model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using the specified step size. We use the entire data set to
   * update the gradient in each iteration.
   *
   * @param input RDD of (label, array of features) pairs.
   * @param stepSize Step size to be used for each iteration of Gradient Descent.
   * @param regParam Regularization parameter.
   * @param numIterations Number of iterations of gradient descent to run.
   * @return a SVMModel which has the weights and offset from training.
   *
   * @note Labels used in SVM should be {0, 1}
   */
  def train(
      input: RDD[UserLabeledPoint],
      numIterations: Int,
      stepSize: Double,
      regParam: Double): LTRSVMModel = {
    train(input, numIterations, stepSize, regParam, 1.0)
  }

  /**
   * Train a SVM model given an RDD of (userId, label, features) pairs. We run a fixed number
   * of iterations of gradient descent using a step size of 1.0. We use the entire data set to
   * update the gradient in each iteration.
   *
   * @param input RDD of (label, array of features) pairs.
   * @param numIterations Number of iterations of gradient descent to run.
   * @return a SVMModel which has the weights and offset from training.
   *
   * @note Labels used in SVM should be {0, 1}
   */
  def train(input: RDD[UserLabeledPoint], numIterations: Int): LTRSVMModel = {
    train(input, numIterations, 1.0, 0.01, 1.0)
  }
}