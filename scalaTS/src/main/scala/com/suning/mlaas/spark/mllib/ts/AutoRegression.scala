package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.regression.LinearRegression
import com.suning.mlaas.spark.mllib.ts.TimeSeriesUtil._
import com.suning.mlaas.spark.mllib.util.{Identifiable, Model, SaveLoad}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class AutoRegression(override val uid: String, inputCol: String, timeCol: String, p: Int,
                     regParam: Double, standardization: Boolean, elasticNetParam: Double,
                     withIntercept: Boolean, meanOut: Boolean)
  extends TSModel(uid, inputCol, timeCol) {

  def this(inputCol: String, timeCol: String, p: Int,
           regParam: Double, standardization: Boolean, elasticNetParam: Double,
           withIntercept: Boolean, meanOut: Boolean) =
    this(Identifiable.randomUID("AutoRegression"), inputCol, timeCol, p, regParam, standardization,
      elasticNetParam, withIntercept, meanOut)

  def this(inputCol: String, timeCol: String, p: Int, regParam: Double, standardization: Boolean,
           elasticNetParam: Double, withIntercept: Boolean) =
    this(Identifiable.randomUID("AutoRegression"), inputCol, timeCol, p, regParam, standardization,
      elasticNetParam, withIntercept, false)


  private var lr_ar: LinearRegression = _

  override def fitImpl(df: DataFrame): this.type = {

    require(p > 0, s"p can not be 0")

    val prefix = if (meanOut) "_meanOut" else ""
    val lag = "_lag_"
    val label = inputCol + prefix + lag + (0)
    val r = 1 to p
    val features = r.map(inputCol + prefix + lag + _).toArray

    val newDF = TimeSeriesUtil.LagCombination(df, inputCol, timeCol, p, lagsOnly = false, meanOut)
      .filter(col(inputCol + prefix + lag + p).isNotNull)

    val maxIter = 1000
    val tol = 1E-6

    newDF.persist()

    lr_ar = LinearRegression(features, label, regParam, withIntercept, standardization,
      elasticNetParam, maxIter, tol)

    lr_ar.fit(newDF)
    newDF.unpersist()

    this
  }

  override def transformImpl(df: DataFrame): DataFrame = {
    require(p > 0, s"p can not be 0")
    val prefix = if (meanOut) "_meanOut" else ""
    val lag = "_lag_"
    val newDF = TimeSeriesUtil.LagCombination(df, inputCol, timeCol, p, lagsOnly = false, meanOut)
      .filter(col(inputCol + prefix + lag + p).isNotNull)
    lr_ar.transform(newDF)
  }

  override def forecast(df: DataFrame, numAhead: Int): List[Double] = {

    require(p > 0, s"p can not be 0")

    if (lr_ar == null) fit(df)

    val newDF = transform(df)

    val prefix = if (meanOut) "_meanOut" else ""
    val lag = "_lag_"
    var listPrediction = newDF.orderBy(desc(timeCol)).select(inputCol + prefix + lag + 0)
      .limit(p).collect().map(_.getDouble(0)).toList

    val meanValue = getDouble(df.select(mean(inputCol)).collect()(0).get(0))

    if (meanOut) {
      listPrediction = tsFitDotProduct(listPrediction, numAhead, p, getIntercept(), getWeights(),
        meanValue = meanValue)
    } else {
      listPrediction = tsFitDotProduct(listPrediction, numAhead, p, getIntercept(), getWeights(),
        meanValue = 0.0)
    }

    val prediction = listPrediction.slice(0, numAhead).reverse
    prediction
  }

  def getIntercept(): Double = {
    lr_ar.getIntercept()
  }

  def getWeights(): Vector = {
    lr_ar.getWeights()
  }

  override def copy(): Model = {
    new AutoRegression(inputCol, timeCol, p, regParam, standardization, elasticNetParam,
      withIntercept, meanOut)
  }

  override def save(path: String): Unit = {
    AutoRegression.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    AutoRegression.saveHDFS(sc, this, path)
  }
}

object AutoRegression extends SaveLoad[AutoRegression] {
  def apply(uid: String, inputCol: String,
            timeCol: String, p: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean, meanOut: Boolean = false):
  AutoRegression = new AutoRegression(uid, inputCol, timeCol, p, regParam, standardization,
    elasticNetParam, withIntercept, meanOut)

  def apply(inputCol: String, timeCol: String, p: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean, meanOut: Boolean):
  AutoRegression = new AutoRegression(inputCol, timeCol, p, regParam, standardization,
    elasticNetParam, withIntercept, meanOut)

  def apply(inputCol: String, timeCol: String, p: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean):
  AutoRegression = new AutoRegression(inputCol, timeCol, p, regParam, standardization,
    elasticNetParam, withIntercept, false)

}
