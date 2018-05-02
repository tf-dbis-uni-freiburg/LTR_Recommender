package org.apache.spark.mllib.optimization

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{ norm, DenseVector => BDV }

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.DenseVector

/**
 * Class used to solve an optimization problem using Gradient Descent.
 * @param gradient Gradient function to be used.
 * @param updater Updater to be used to update weights after every iteration.
 */
class LTRGradientDescent private[spark] (private var gradient: Gradient, private var updater: Updater) {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private var convergenceTol: Double = 0.001

  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    require(
      step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }

  /**
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(
      fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    require(
      iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    require(
      regParam >= 0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }

  /**
   * Set the convergence tolerance. Default 0.001
   * convergenceTol is a condition which decides iteration termination.
   * The end of iteration is decided based on below logic.
   *
   *  - If the norm of the new solution vector is greater than 1, the diff of solution vectors
   *    is compared to relative tolerance which means normalizing by the norm of
   *    the new solution vector.
   *  - If the norm of the new solution vector is less than or equal to 1, the diff of solution
   *    vectors is compared to absolute tolerance which is not normalizing.
   *
   * Must be between 0.0 and 1.0 inclusively.
   */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(
      tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for SGD.
   */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
   * Runs gradient descent on the given training data.
   * @param data training data
   * @param initialWeights initial weights
   * @return solution vector
   */
  def optimize(data: RDD[(Int, Double, Vector)], initialWeights: Map[Int, Vector]): Map[Int, Vector] = {
    val (weights, _) = LTRGradientDescent.runMiniBatchSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights,
      convergenceTol)
    weights
  }

}

object LTRGradientDescent extends Logging {
  /**
   * Run stochastic gradient descent (SGD) in parallel using mini batches.
   * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
   * in order to compute a gradient estimate.
   * Sampling, and averaging the subgradients over this subset is performed using one standard
   * spark map-reduce in each iteration.
   *
   * @param data Input data for SGD. RDD of the set of data examples, each of
   *             the form (label, [feature values]).
   * @param gradient Gradient object (used to compute the gradient of the loss function of
   *                 one single data example)
   * @param updater Updater function to actually perform a gradient step in a given direction.
   * @param stepSize initial step size for the first step
   * @param numIterations number of iterations that SGD should be run.
   * @param regParam regularization parameter
   * @param miniBatchFraction fraction of the input data set that should be used for
   *                          one iteration of SGD. Default value 1.0.
   * @param convergenceTol Minibatch iteration will end before numIterations if the relative
   *                       difference between the current weight and the previous weight is less
   *                       than this value. In measuring convergence, L2 norm is calculated.
   *                       Default value 0.001. Must be between 0.0 and 1.0 inclusively.
   * @return A tuple containing two elements. The first element is a column matrix containing
   *         weights for every feature, and the second element is an array containing the
   *         stochastic loss computed for every iteration.
   */
  def runMiniBatchSGD(
    data: RDD[(Int, Double, Vector)],
    gradient: Gradient,
    updater: Updater,
    stepSize: Double,
    numIterations: Int,
    regParam: Double,
    miniBatchFraction: Double,
    initialWeights: Map[Int, Vector],
    // return map - weight vector for each user, array - loss for each iteration, for each user in a map
    convergenceTol: Double): (Map[Int, Vector], Array[collection.mutable.Map[Int, Double]]) = {
    
    // convergenceTol should be set with non minibatch settings
    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      println("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }

    if (numIterations * miniBatchFraction < 1.0) {
      println("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
        s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
    }

    val stochasticLossHistory = new ArrayBuffer[scala.collection.mutable.Map[Int, Double]](numIterations)

    // Record previous weight and current one to calculate solution vector difference
    var previousWeights: Option[Map[Int, Vector]] = None
    var currentWeights: Option[Map[Int, Vector]] = None
    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    // TODO here we can check if there is enough examples for each user, not only the full data set count
    if (numExamples == 0) {
      println("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }
    // TODO check here for each user not only the full data set count
    if (numExamples * miniBatchFraction < 1) {
      println("The miniBatchFraction is too small")
    }

    // Initialize weights as a column vector
    // convert vectors to dense vectors
    var weights = collection.mutable.Map[Int, Vector]()
    initialWeights foreach {
      case (userId, vector) =>
        weights.put(userId, Vectors.dense(vector.toArray))
    }
    // num features
    var nFeatures = data.map(_._3.size).first()
   
    /**
     * For the first iteration, the regVal will be initialized as sum of weight squares
     * if it's L2 updater; for L1 updater, the same logic is followed.
     */
    var regVal = scala.collection.mutable.Map[Int, Double]()
    weights foreach {
      case (userId, userWeights) =>
        var userRegVal = updater.compute(userWeights, Vectors.zeros(userWeights.size), 0, 1, regParam)._2
        regVal.put(userId, userRegVal)
    }
    var converged = scala.collection.mutable.Map[Int, Boolean]()
    // initialize converged with false for each user
    weights foreach {
      case (userId, userWeights) =>
        // indicates whether converged based on convergenceTol
        converged.update(userId, false)
    }   
    var i = 1
    while (i <= numIterations) {
      // TODO add the convertage - check for every user if the weights already converged for it
      val bcWeights = data.context.broadcast(weights)
      
      // compute and sum up the subgradients on this subset (this is one map-reduce)
      // gradientSum - map[int, vector] for each user
      var zeroGradientSum = collection.mutable.Map[Int, breeze.linalg.DenseVector[Double]]()
      weights.foreach {
        case (userId, vector) =>
          zeroGradientSum.put(userId, BDV.zeros[Double](nFeatures))
      }
      
      // lossSum - map[int, double] for each user
      var zeroLossSum = collection.mutable.Map[Int, Double]()
      weights.foreach {
        case (userId, vector) =>
          zeroLossSum.put(userId, 0.0)
      }
     
      // zeroSamplesPerUser - map[int, int] for each user
      var zeroSamplesCountPerUser = collection.mutable.Map[Int, Int]()
      weights.foreach {
        case (userId, vector) =>
          zeroSamplesCountPerUser.put(userId, 0)
      }
      
      val (gradientSum, lossSum, samplesCountPerUser) = data
        .treeAggregate((zeroGradientSum, zeroLossSum, zeroSamplesCountPerUser))(
          seqOp = (c, v) => {
            // c: (grad, loss, countPerUser), v: (userId, label, features)
            val localUserWeight = bcWeights.value.get(v._1)
            // gradient: Vector, loss: Double
            val l = gradient.compute(v._3, v._2, localUserWeight.get) 
            // update user gradient
            var localUserGradient = c._1.get(v._1).get
            var currentGradientSum = l._1.asBreeze
            localUserGradient += currentGradientSum
            c._1.put(v._1, localUserGradient)
            // update user loss
            val localUserLoss = c._2.get(v._1).get
            c._2.put(v._1, localUserLoss + l._2)
            // update how many examples we encounter per user
            val samplesCountPerUser =  c._3.get(v._1).get
            c._3.put(v._1, samplesCountPerUser + 1)
            (c._1, c._2, c._3)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, countPerUser)
            // merge two maps that contains gradients
            c2._1 foreach {
              case (userId, grad) =>
                // if this userId exists in the gradientMap of c1, update it by
                // gradient calculate in c2
                // otherwise, just add it to c1 gradient map
                if (c1._1.contains(userId)) {
                  val c1GradVector = c1._1.get(userId).get
                  c1GradVector += grad
                  c1._1.put(userId, c1GradVector)
                } else {
                  c1._1.put(userId, grad)
                }
            }
            // merge two maps that contains loss
            c2._2 foreach {
              case (userId, loss) =>
                val c1loss = c1._2.get(userId).get
                c1._2.put(userId, c1loss + loss)
            }
            
            c2._3 foreach {
              case (userId, count) =>
                if (c1._3.contains(userId)) {
                  val localCount = c1._3.get(userId).get
                  c1._3.put(userId, localCount + count)
                } else {
                  c1._3.put(userId, count)
                }
            }
            
            (c1._1, c1._2, c1._3)
          })
  
      if (numExamples > 0) {
        /**
         * lossSum is computed using the weights from the previous iteration
         * and regVal is the regularisation value computed in the previous iteration as well.
         */
        var currentStochasticLossHistory = collection.mutable.Map[Int, Double]()
        lossSum foreach {
          case (userId, userLossSum) =>
            val userRegVal = regVal.get(userId).getOrElse(0.0)
            val userSampleSize = samplesCountPerUser.get(userId).get
            currentStochasticLossHistory.put(userId, userLossSum / userSampleSize + userRegVal)
        }
        stochasticLossHistory += currentStochasticLossHistory

        gradientSum foreach {
          case (userId, gradientUserSum) =>
            val userWeight = weights.get(userId).get
            val userSampleSize = samplesCountPerUser.get(userId).get
            val update = updater.compute(userWeight, Vectors.fromBreeze(gradientUserSum / userSampleSize.toDouble), stepSize, i, regParam)
            weights.put(userId, update._1)
            regVal.put(userId, update._2)
        }
     
        // update converged map
        previousWeights = currentWeights
        currentWeights = Some(weights.toMap)
        if (previousWeights != None && currentWeights != None) {
          // TODO think if previousWeights and currentWeights has to be the same
          previousWeights.get foreach {
            case (userId, userPreviousWeights) =>
              val userCurrentWeights = currentWeights.get(userId)
              if (userCurrentWeights != None) {
                val userConverged = isConverged(userPreviousWeights, userCurrentWeights, convergenceTol)
                converged.put(userId, userConverged)
              }
          }
        }
      } else {
        println(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      i += 1
    }

    println("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights.toMap, stochasticLossHistory.toArray)

  }

  /**
   * Alias of `runMiniBatchSGD` with convergenceTol set to default value of 0.001.
   */
  def runMiniBatchSGD(
    data: RDD[(Int, Double, Vector)],
    gradient: Gradient,
    updater: Updater,
    stepSize: Double,
    numIterations: Int,
    regParam: Double,
    miniBatchFraction: Double,
    initialWeights: Map[Int, Vector]): (Map[Int, Vector], Array[collection.mutable.Map[Int, Double]]) =
    LTRGradientDescent.runMiniBatchSGD(data, gradient, updater, stepSize, numIterations,
      regParam, miniBatchFraction, initialWeights, 0.001)

  private def isConverged(
    previousWeights: Vector,
    currentWeights: Vector,
    convergenceTol: Double): Boolean = {
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.asBreeze.toDenseVector
    val currentBDV = currentWeights.asBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = norm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }

}
