/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.regression

import org.apache.spark.SparkException
import org.apache.spark.annotation.{ DeveloperApi, Since }
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.mllib.optimization.LTRGradientDescent
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.util.MLUtils._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.storage.StorageLevel
/**
 *
 * LTRGeneralizedLinearModel (GLM) represents a model trained using
 * GeneralizedLinearAlgorithm. GLMs consist of a map with weight vectors and
 * am map with intercepts.
 *
 * @param weights Weights computed for every feature.
 * @param intercept Intercept computed for this model.
 *
 */
abstract class LTRGeneralizedLinearModel(
  val weights: Map[Int, Vector],
  val intercept: Map[Int, Double])
  extends Serializable {

  /**
   * Predict the result given a data point and the weights learned.
   *
   * @param dataMatrix Row vector containing the features for this data point
   * @param weightMatrix Column vector containing the weights of the model
   * @param intercept Intercept of the model.
   */
  protected def predictPoint(userId: Int, dataMatrix: Vector, weightMatrix: Map[Int, Vector], intercept: Map[Int, Double]): Double

  /**
   * Predict values for the given data set using the model trained.
   *
   * @param testData RDD representing data points to be predicted (userId, features)
   * @return RDD[Double] where each entry contains the corresponding prediction
   *
   */
  // TODO change here in input parameters
  def predict(testData: RDD[(Int, Vector)]): RDD[Double] = {
    // A small optimization to avoid serializing the entire model. Only the weightsMatrix
    // and intercept is needed.
    val localWeights = weights
    val bcWeights = testData.context.broadcast(localWeights)
    val localIntercept = intercept
    // TODO change here
    testData.mapPartitions { iter =>
      val w = bcWeights.value
      iter.map {
        case (userId, v) => predictPoint(userId, v, w, localIntercept)
      }
    }
  }

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param testData array representing a single data point
   * @return Double prediction from the trained model
   *
   */
  @Since("1.0.0")
  def predict(userId: Int, testData: Vector): Double = {
    predictPoint(userId, testData, weights, intercept)
  }

  /**
   * Print a summary of the model.
   */
  override def toString: String = {
    s"${this.getClass.getName}: intercept = ${intercept}, numFeatures = ${weights.size}"
  }
}

/**
 * LTRGeneralizedLinearAlgorithm implements methods to train a Generalized Linear Model (GLM).
 * This class should be extended with an Optimizer to create a new GLM.
 *
 */
abstract class LTRGeneralizedLinearAlgorithm[M <: LTRGeneralizedLinearModel]
  extends Logging with Serializable {

  /**
   * The optimizer to solve the problem.
   *
   */
  def optimizer: LTRGradientDescent

  /** Whether to add intercept (default: false). */
  protected var addIntercept: Boolean = false

  /**
   * Whether to perform feature scaling before model training to reduce the condition numbers
   * which can significantly help the optimizer converging faster. The scaling correction will be
   * translated back to resulting model weights, so it's transparent to users.
   * Note: This technique is used in both libsvm and glmnet packages. Default false.
   */
  private[mllib] var useFeatureScaling = false

  /**
   * The dimension of training features.
   *
   */
  def getNumFeatures: Int = this.numFeatures

  /**
   * The dimension of training features.
   */
  protected var numFeatures: Int = -1

  /**
   * Set if the algorithm should use feature scaling to improve the convergence during optimization.
   */
  private[mllib] def setFeatureScaling(useFeatureScaling: Boolean): this.type = {
    this.useFeatureScaling = useFeatureScaling
    this
  }

  /**
   * Create a model given the weights and intercept
   */
  protected def createModel(weights: Map[Int, Vector], intercept: Map[Int, Double]): M

  /**
   * Get if the algorithm uses addIntercept
   *
   */
  def isAddIntercept: Boolean = this.addIntercept

  /**
   * Set if the algorithm should add an intercept. Default false.
   * We set the default to false because adding the intercept will cause memory allocation.
   *
   */
  def setIntercept(addIntercept: Boolean): this.type = {
    this.addIntercept = addIntercept
    this
  }

  /**
   * Generate the initial weights when the user does not supply them
   */
  protected def generateInitialWeights(input: RDD[UserLabeledPoint]): collection.mutable.Map[Int, Vector] = {
    if (numFeatures < 0) {
      numFeatures = input.map(_.features.size).first()
    }

    val mutableInitialWeights = collection.mutable.Map[Int, Vector]()
    val userIds = input.map(x => x.userId)
   
    userIds.distinct().collect().foreach(
      userId => {
        mutableInitialWeights.put(userId, Vectors.zeros(numFeatures))
      })
    mutableInitialWeights
  }

  /**
   * Run the algorithm with the configured parameters on an input
   * RDD of UserLabeledPoint entries.
   *
   */
  def run(input: RDD[UserLabeledPoint]): M = {
    run(input, generateInitialWeights(input))
  }

  /**
   * Run the algorithm with the configured parameters on an input RDD
   * of UserLabeledPoint entries starting from the initial weights provided.
   *
   */
  def run(input: RDD[UserLabeledPoint], initialWeights: collection.mutable.Map[Int, Vector]): M = {
    if (numFeatures < 0) {
      numFeatures = input.map(_.features.size).first()
    }

    if (input.getStorageLevel == StorageLevel.NONE) {
      println("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
    
    val data = input.map(lp => (lp.userId, lp.label, lp.features))
   
    /**
     * TODO: For better convergence, in logistic regression, the intercepts should be computed
     * from the prior probability distribution of the outcomes; for linear regression,
     * the intercept should be set as the average of response.
     */
    var initialWeightsWithIntercept = initialWeights.toMap
    val weightsWithIntercept = optimizer.optimize(data, initialWeightsWithIntercept)
    
    // intercept will be map[int, double] key - userId, value - intercept for this user
    val intercept = {
      val intercept = collection.mutable.Map[Int, Double]()
      weightsWithIntercept foreach {
        case (userId, weightsWithIntercept) =>
          intercept.update(userId, 0.0)
      }
      intercept
    }

    // remove the intercept from the weights
    // weights will be map[int, vector], key - userId, value - weight vector for this user
   var weights = weightsWithIntercept

    // Warn at the end of the run as well, for increased visibility.
    if (input.getStorageLevel == StorageLevel.NONE) {
      println("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Unpersist cached data
    if (data.getStorageLevel != StorageLevel.NONE) {
      data.unpersist(false)
    }
    // convert them to immutable
    val finalWeights = weights.toMap
    val finalIntercept = intercept.toMap
    createModel(finalWeights, finalIntercept)
  }

  def appendBiasToMap(map: collection.mutable.Map[Int, Vector]): scala.collection.immutable.Map[Int, Vector] = {
    map foreach {
      case (userId, vector) =>
        vector match {
          case dv: DenseVector =>
            val inputValues = dv.values
            val inputLength = inputValues.length
            val outputValues = Array.ofDim[Double](inputLength + 1)
            System.arraycopy(inputValues, 0, outputValues, 0, inputLength)
            outputValues(inputLength) = 1.0
            map.put(userId, Vectors.dense(outputValues))
          case sv: SparseVector =>
            val inputValues = sv.values
            val inputIndices = sv.indices
            val inputValuesLength = inputValues.length
            val dim = sv.size
            val outputValues = Array.ofDim[Double](inputValuesLength + 1)
            val outputIndices = Array.ofDim[Int](inputValuesLength + 1)
            System.arraycopy(inputValues, 0, outputValues, 0, inputValuesLength)
            System.arraycopy(inputIndices, 0, outputIndices, 0, inputValuesLength)
            outputValues(inputValuesLength) = 1.0
            outputIndices(inputValuesLength) = dim
            map.put(userId, Vectors.sparse(dim + 1, outputIndices, outputValues))
          case _ => throw new IllegalArgumentException(s"Do not support vector type ${vector.getClass}")
        }
    }
    map.toMap
  }

}
