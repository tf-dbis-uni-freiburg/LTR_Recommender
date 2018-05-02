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

package org.apache.spark.mllib.api.python

import java.util.{ ArrayList => JArrayList, List => JList, Map => JMap }
import scala.collection.JavaConverters._

import org.apache.spark.api.java.{ JavaRDD, JavaSparkContext }
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.storage.StorageLevel

/**
 * The Java stubs necessary for the Python mllib bindings. It is called by Py4J on the Python side.
 */
private[python] class LTRPythonMLLibAPI extends PythonMLLibAPI {

  /**
   * Java stub for Python mllib SVMWithSGD.train()
   */
  def trainLTRSVMModelWithSGD(
    data: JavaRDD[UserLabeledPoint],
    numIterations: Int,
    stepSize: Double,
    regParam: Double,
    miniBatchFraction: Double,
    regType: String,
    intercept: Boolean,
    validateData: Boolean,
    convergenceTol: Double): JList[Object] = {
    val SVMAlg = new LTRSVMWithSGD()
    SVMAlg.setIntercept(intercept)
      .setValidateData(validateData)
    SVMAlg.optimizer
      .setNumIterations(numIterations)
      .setRegParam(regParam)
      .setStepSize(stepSize)
      .setMiniBatchFraction(miniBatchFraction)
      .setConvergenceTol(convergenceTol)
    SVMAlg.optimizer.setUpdater(getUpdaterFromString(regType))
    this.trainRegressionModel(
      SVMAlg,
      data) 
  }

  private def trainRegressionModel(
    learner: LTRGeneralizedLinearAlgorithm[_ <: LTRGeneralizedLinearModel],
    data: JavaRDD[UserLabeledPoint]): JList[Object] = {
    try {
      val model =  learner.run(data.rdd.persist(StorageLevel.MEMORY_AND_DISK))
      List(model.weights, model.intercept).map(_.asInstanceOf[Object]).asJava
    } finally {
      data.rdd.unpersist(blocking = false)
    }
  }
}

