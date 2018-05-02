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

import java.io.OutputStream
import java.nio.{ ByteBuffer, ByteOrder }
import java.nio.charset.StandardCharsets
import java.util.{ ArrayList => JArrayList, List => JList, Map => JMap }

import scala.collection.JavaConverters._
import scala.language.existentials
import scala.reflect.ClassTag

import net.razorvine.pickle._
import org.apache.spark.api.java.{ JavaRDD, JavaSparkContext }
import org.apache.spark.api.python.SerDeUtil
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.mllib.feature._
import org.apache.spark.mllib.fpm.{ FPGrowth, FPGrowthModel, PrefixSpan }
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.random.{ RandomRDDs => RG }
import org.apache.spark.mllib.recommendation._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.stat.{ KernelDensity, MultivariateStatisticalSummary, Statistics }
import org.apache.spark.mllib.stat.correlation.CorrelationNames
import org.apache.spark.mllib.stat.distribution.MultivariateGaussian
import org.apache.spark.mllib.stat.test.{ ChiSqTestResult, KolmogorovSmirnovTestResult }
import org.apache.spark.mllib.tree.{ DecisionTree, GradientBoostedTrees, RandomForest }
import org.apache.spark.mllib.tree.configuration.{ Algo, BoostingStrategy, Strategy }
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.mllib.tree.loss.Losses
import org.apache.spark.mllib.tree.model.{
  DecisionTreeModel,
  GradientBoostedTreesModel,
  RandomForestModel
}
import org.apache.spark.mllib.util.{ LinearDataGenerator, MLUtils }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ DataFrame, Row, SparkSession }
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils

/**
 * Basic SerDe utility class.
 */
private[spark] abstract class LTRSerDeBase {

  val PYSPARK_PACKAGE: String
  def initialize(): Unit

  /**
   * Base class used for pickle
   */
  private[spark] abstract class BasePickler[T: ClassTag]
    extends IObjectPickler with IObjectConstructor {

    private val cls = implicitly[ClassTag[T]].runtimeClass
    private val module = PYSPARK_PACKAGE + "." + cls.getName.split('.')(4)
    private val name = cls.getSimpleName

    // register this to Pickler and Unpickler
    def register(): Unit = {
      Pickler.registerCustomPickler(this.getClass, this)
      Pickler.registerCustomPickler(cls, this)
      if (name == "UserLabeledPoint") {
        Unpickler.registerConstructor("LTR_SVM_spark2", name, this)
      } else {
        Unpickler.registerConstructor(module, name, this)
      }
    }

    def pickle(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      if (obj == this) {
        out.write(Opcodes.GLOBAL)
        out.write((module + "\n" + name + "\n").getBytes(StandardCharsets.UTF_8))
      } else {
        pickler.save(this) // it will be memorized by Pickler
        saveState(obj, out, pickler)
        out.write(Opcodes.REDUCE)
      }
    }

    private[python] def saveObjects(out: OutputStream, pickler: Pickler, objects: Any*) = {
      if (objects.length == 0 || objects.length > 3) {
        out.write(Opcodes.MARK)
      }
      objects.foreach(pickler.save)
      val code = objects.length match {
        case 1 => Opcodes.TUPLE1
        case 2 => Opcodes.TUPLE2
        case 3 => Opcodes.TUPLE3
        case _ => Opcodes.TUPLE
      }
      out.write(code)
    }

    protected def getBytes(obj: Object): Array[Byte] = {
      if (obj.getClass.isArray) {
        obj.asInstanceOf[Array[Byte]]
      } else {
        // This must be ISO 8859-1 / Latin 1, not UTF-8, to interoperate correctly
        obj.asInstanceOf[String].getBytes(StandardCharsets.ISO_8859_1)
      }
    }

    private[python] def saveState(obj: Object, out: OutputStream, pickler: Pickler)
  }

  def dumps(obj: AnyRef): Array[Byte] = {
    obj match {
      // Pickler in Python side cannot deserialize Scala Array normally. See SPARK-12834.
      case array: Array[_] => new Pickler().dumps(array.toSeq.asJava)
      case _ => new Pickler().dumps(obj)
    }
  }

  def loads(bytes: Array[Byte]): AnyRef = {
    new Unpickler().loads(bytes)
  }

  /* convert object into Tuple */
  def asTupleRDD(rdd: RDD[Array[Any]]): RDD[(Int, Int)] = {
    rdd.map(x => (x(0).asInstanceOf[Int], x(1).asInstanceOf[Int]))
  }

  /* convert RDD[Tuple2[,]] to RDD[Array[Any]] */
  def fromTuple2RDD(rdd: RDD[(Any, Any)]): RDD[Array[Any]] = {
    rdd.map(x => Array(x._1, x._2))
  }

  /**
   * Convert an RDD of Java objects to an RDD of serialized Python objects, that is usable by
   * PySpark.
   */
  def javaToPython(jRDD: JavaRDD[Any]): JavaRDD[Array[Byte]] = {
    jRDD.rdd.mapPartitions { iter =>
      initialize() // let it called in executor
      new SerDeUtil.AutoBatchedPickler(iter)
    }
  }

  /**
   * Convert an RDD of serialized Python objects to RDD of objects, that is usable by PySpark.
   */
  def pythonToJava(pyRDD: JavaRDD[Array[Byte]], batched: Boolean): JavaRDD[Any] = {
    val customRdd = pyRDD.rdd.mapPartitions { iter =>
      initialize() // let it called in executor
      val unpickle = new Unpickler
      iter.flatMap { row =>
        val obj = unpickle.loads(row)
        if (batched) {
          obj match {
            case list: JArrayList[_] =>
              list.asScala
            case arr: Array[_] =>
              arr
          }
        } else {
          Seq(obj)
        }
      }
    }
    customRdd.toJavaRDD()
  }
}

/**
 * SerDe utility functions for PythonMLLibAPI.
 */
private[spark] object LTRSerDe extends LTRSerDeBase with Serializable {

  override val PYSPARK_PACKAGE = "pyspark.mllib"

  // Pickler for DenseVector
  private[python] class DenseVectorPickler extends BasePickler[DenseVector] {

    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      val vector: DenseVector = obj.asInstanceOf[DenseVector]
      val bytes = new Array[Byte](8 * vector.size)
      val bb = ByteBuffer.wrap(bytes)
      bb.order(ByteOrder.nativeOrder())
      val db = bb.asDoubleBuffer()
      db.put(vector.values)

      out.write(Opcodes.BINSTRING)
      out.write(PickleUtils.integer_to_bytes(bytes.length))
      out.write(bytes)
      out.write(Opcodes.TUPLE1)
    }

    def construct(args: Array[Object]): Object = {
      require(args.length == 1)
      if (args.length != 1) {
        throw new PickleException("should be 1")
      }
      val bytes = getBytes(args(0))
      val bb = ByteBuffer.wrap(bytes, 0, bytes.length)
      bb.order(ByteOrder.nativeOrder())
      val db = bb.asDoubleBuffer()
      val ans = new Array[Double](bytes.length / 8)
      db.get(ans)
      Vectors.dense(ans)
    }
  }

  //  // Pickler for DenseMatrix
  //  private[python] class DenseMatrixPickler extends BasePickler[DenseMatrix] {
  //
  //    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
  //      val m: DenseMatrix = obj.asInstanceOf[DenseMatrix]
  //      val bytes = new Array[Byte](8 * m.values.length)
  //      val order = ByteOrder.nativeOrder()
  //      val isTransposed = if (m.isTransposed) 1 else 0
  //      ByteBuffer.wrap(bytes).order(order).asDoubleBuffer().put(m.values)
  //
  //      out.write(Opcodes.MARK)
  //      out.write(Opcodes.BININT)
  //      out.write(PickleUtils.integer_to_bytes(m.numRows))
  //      out.write(Opcodes.BININT)
  //      out.write(PickleUtils.integer_to_bytes(m.numCols))
  //      out.write(Opcodes.BINSTRING)
  //      out.write(PickleUtils.integer_to_bytes(bytes.length))
  //      out.write(bytes)
  //      out.write(Opcodes.BININT)
  //      out.write(PickleUtils.integer_to_bytes(isTransposed))
  //      out.write(Opcodes.TUPLE)
  //    }
  //
  //    def construct(args: Array[Object]): Object = {
  //      if (args.length != 4) {
  //        throw new PickleException("should be 4")
  //      }
  //      val bytes = getBytes(args(2))
  //      val n = bytes.length / 8
  //      val values = new Array[Double](n)
  //      val order = ByteOrder.nativeOrder()
  //      ByteBuffer.wrap(bytes).order(order).asDoubleBuffer().get(values)
  //      val isTransposed = args(3).asInstanceOf[Int] == 1
  //      new DenseMatrix(args(0).asInstanceOf[Int], args(1).asInstanceOf[Int], values, isTransposed)
  //    }
  //  }

  // TODO remove it
  //  // Pickler for SparseMatrix
  //  private[python] class SparseMatrixPickler extends BasePickler[SparseMatrix] {
  //
  //    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
  //      val s = obj.asInstanceOf[SparseMatrix]
  //      val order = ByteOrder.nativeOrder()
  //
  //      val colPtrsBytes = new Array[Byte](4 * s.colPtrs.length)
  //      val indicesBytes = new Array[Byte](4 * s.rowIndices.length)
  //      val valuesBytes = new Array[Byte](8 * s.values.length)
  //      val isTransposed = if (s.isTransposed) 1 else 0
  //      ByteBuffer.wrap(colPtrsBytes).order(order).asIntBuffer().put(s.colPtrs)
  //      ByteBuffer.wrap(indicesBytes).order(order).asIntBuffer().put(s.rowIndices)
  //      ByteBuffer.wrap(valuesBytes).order(order).asDoubleBuffer().put(s.values)
  //
  //      out.write(Opcodes.MARK)
  //      out.write(Opcodes.BININT)
  //      out.write(PickleUtils.integer_to_bytes(s.numRows))
  //      out.write(Opcodes.BININT)
  //      out.write(PickleUtils.integer_to_bytes(s.numCols))
  //      out.write(Opcodes.BINSTRING)
  //      out.write(PickleUtils.integer_to_bytes(colPtrsBytes.length))
  //      out.write(colPtrsBytes)
  //      out.write(Opcodes.BINSTRING)
  //      out.write(PickleUtils.integer_to_bytes(indicesBytes.length))
  //      out.write(indicesBytes)
  //      out.write(Opcodes.BINSTRING)
  //      out.write(PickleUtils.integer_to_bytes(valuesBytes.length))
  //      out.write(valuesBytes)
  //      out.write(Opcodes.BININT)
  //      out.write(PickleUtils.integer_to_bytes(isTransposed))
  //      out.write(Opcodes.TUPLE)
  //    }
  //
  //    def construct(args: Array[Object]): Object = {
  //      if (args.length != 6) {
  //        throw new PickleException("should be 6")
  //      }
  //      val order = ByteOrder.nativeOrder()
  //      val colPtrsBytes = getBytes(args(2))
  //      val indicesBytes = getBytes(args(3))
  //      val valuesBytes = getBytes(args(4))
  //      val colPtrs = new Array[Int](colPtrsBytes.length / 4)
  //      val rowIndices = new Array[Int](indicesBytes.length / 4)
  //      val values = new Array[Double](valuesBytes.length / 8)
  //      ByteBuffer.wrap(colPtrsBytes).order(order).asIntBuffer().get(colPtrs)
  //      ByteBuffer.wrap(indicesBytes).order(order).asIntBuffer().get(rowIndices)
  //      ByteBuffer.wrap(valuesBytes).order(order).asDoubleBuffer().get(values)
  //      val isTransposed = args(5).asInstanceOf[Int] == 1
  //      new SparseMatrix(
  //        args(0).asInstanceOf[Int], args(1).asInstanceOf[Int], colPtrs, rowIndices, values,
  //        isTransposed)
  //    }
  //  }

  // Pickler for SparseVector
  private[python] class SparseVectorPickler extends BasePickler[SparseVector] {

    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      val v: SparseVector = obj.asInstanceOf[SparseVector]
      val n = v.indices.length
      val indiceBytes = new Array[Byte](4 * n)
      val order = ByteOrder.nativeOrder()
      ByteBuffer.wrap(indiceBytes).order(order).asIntBuffer().put(v.indices)
      val valueBytes = new Array[Byte](8 * n)
      ByteBuffer.wrap(valueBytes).order(order).asDoubleBuffer().put(v.values)

      out.write(Opcodes.BININT)
      out.write(PickleUtils.integer_to_bytes(v.size))
      out.write(Opcodes.BINSTRING)
      out.write(PickleUtils.integer_to_bytes(indiceBytes.length))
      out.write(indiceBytes)
      out.write(Opcodes.BINSTRING)
      out.write(PickleUtils.integer_to_bytes(valueBytes.length))
      out.write(valueBytes)
      out.write(Opcodes.TUPLE3)
    }

    def construct(args: Array[Object]): Object = {
      if (args.length != 3) {
        throw new PickleException("should be 3")
      }
      val size = args(0).asInstanceOf[Int]
      val indiceBytes = getBytes(args(1))
      val valueBytes = getBytes(args(2))
      val n = indiceBytes.length / 4
      val indices = new Array[Int](n)
      val values = new Array[Double](n)
      if (n > 0) {
        val order = ByteOrder.nativeOrder()
        ByteBuffer.wrap(indiceBytes).order(order).asIntBuffer().get(indices)
        ByteBuffer.wrap(valueBytes).order(order).asDoubleBuffer().get(values)
      }
      new SparseVector(size, indices, values)
    }
  }

  // Pickler for MLlib LabeledPoint
  private[python] class UserLabeledPointPickler extends BasePickler[UserLabeledPoint] {

    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      val point: UserLabeledPoint = obj.asInstanceOf[UserLabeledPoint]
      saveObjects(out, pickler, point.userId, point.label, point.features)
    }

    def construct(args: Array[Object]): Object = {
      if (args.length != 3) {
        throw new PickleException("should be 3")
      }
      new UserLabeledPoint(args(0).asInstanceOf[Int], args(1).asInstanceOf[Double], args(2).asInstanceOf[Vector])
    }
  }

  // Pickler for MLlib LabeledPoint
  private[python] class LabeledPointPickler extends BasePickler[LabeledPoint] {

    def saveState(obj: Object, out: OutputStream, pickler: Pickler): Unit = {
      val point: LabeledPoint = obj.asInstanceOf[LabeledPoint]
      saveObjects(out, pickler, point.label, point.features)
    }

    def construct(args: Array[Object]): Object = {
      if (args.length != 2) {
        throw new PickleException("should be 2")
      }
      new LabeledPoint(args(0).asInstanceOf[Double], args(1).asInstanceOf[Vector])
    }
  }

  var initialized = false
  // This should be called before trying to serialize any above classes
  // In cluster mode, this should be put in the closure
  override def initialize(): Unit = {
    SerDeUtil.initialize()
    synchronized {
      if (!initialized) {
        new UserLabeledPointPickler().register()
        new DenseVectorPickler().register()
        //        new DenseMatrixPickler().register()
        //        new SparseMatrixPickler().register()
        new SparseVectorPickler().register()
        new LabeledPointPickler().register()
        initialized = true
      }
    }
  }
  // will not called in Executor automatically
  initialize()
}
