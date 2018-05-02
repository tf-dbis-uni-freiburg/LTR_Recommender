package org.apache.spark.mllib.regression
import scala.beans.BeanInfo

import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.NumericParser
import org.apache.spark.SparkException

/**
 * Class that represents user id, the features and labels of a data point.
 *
 * @param userId id of an user to which this data point belongs
 * @param label Label for this data point.
 * @param features List of features for this data point.
 */
@BeanInfo
case class UserLabeledPoint(
    userId: Int,
    label: Double,
    features: Vector) {
  override def toString: String = {
    s"($userId,$label,$features)"
  }
}

/**
 * Parser for [[org.apache.spark.mllib.regression.UserLabeledPoint]].
 *
 */
object UserLabeledPoint {
  /**
   * Parses a string resulted from `UserLabeledPoint#toString` into
   * an [[org.apache.spark.mllib.regression.UserLabeledPoint]].
   *
   */
  def parse(s: String): UserLabeledPoint = {
    if (s.startsWith("(")) {
      NumericParser.parse(s) match {
        case Seq(userId:Int, label: Double, numeric: Any) =>
          UserLabeledPoint(userId, label, Vectors.parseNumeric(numeric))
        case other =>
          throw new SparkException(s"Cannot parse $other.")
      }
    } else { // dense format used before v1.0
      val parts = s.split(',')
      val userId = java.lang.Integer.parseInt(parts(0))
      val label = java.lang.Double.parseDouble(parts(1))
      val features = Vectors.dense(parts(2).trim().split(' ').map(java.lang.Double.parseDouble))
      UserLabeledPoint(userId, label, features)
    }
  }
}
