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
  
class MovingAverage(override val uid: String, inputCol: String, timeCol: String, q: Int,
                    regParam: Double, standardization: Boolean, elasticNetParam: Double,
                    withIntercept: Boolean, meanOut: Boolean)
  extends TSModel(uid, inputCol, timeCol) {

  def this(inputCol: String, timeCol: String, q: Int,
           regParam: Double, standardization: Boolean = true, elasticNetParam: Double,
           withIntercept: Boolean = false, meanOut: Boolean) =
    this(Identifiable.randomUID("MovingAverage"), inputCol, timeCol, q, regParam, standardization,
      elasticNetParam, withIntercept, meanOut)

  private var lr_ma: LinearRegression = _

  private val residual = "residual"
  private val prediction = "prediction"
  private val label = "label"
  private val lag = "_lag_"
  private val maLabel = "maLabel"


  override def fitImpl(df: DataFrame): this.type = {
    require(q > 0, s"q can not be 0")
    val arModel = AutoRegression(inputCol, timeCol, q,
      regParam, standardization, elasticNetParam, withIntercept, meanOut)

    arModel.fit(df)

    val residualDF = arModel.transform(df)
      .withColumn(residual, -col(prediction) + col(label))
      .drop(prediction)
    //    residualDF.show()

    //    val newDF = TimeSeriesUtil.LagCombinationMA(residualDF, residual, timeCol, label, q)
    //      .filter(col(residual + lag + q).isNotNull)
    //      .withColumnRenamed(label, maLabel)

    val newDF = TimeSeriesUtil.LagCombinationMA(residualDF, inputCol, residual, timeCol, label, q)
      .filter(col(residual + lag + q).isNotNull)
      .withColumnRenamed(inputCol + lag + "0", maLabel)
    //    newDF.show()

    val features = (1 to q toArray).map(
      residual + lag + _)

    val maxIter = 1000
    val tol = 1E-6

    newDF.persist()

    lr_ma = LinearRegression(features, maLabel, regParam, withIntercept, standardization,
      elasticNetParam, maxIter, tol)

    lr_ma.fit(newDF)

    newDF.unpersist()

    this

  }

  override def transformImpl(df: DataFrame): DataFrame = {
    require(q > 0, s"q can not be 0")

    val arModel = AutoRegression(inputCol, timeCol, q, regParam, standardization, elasticNetParam,
      withIntercept)
    val features = "features"
    arModel.fit(df)
    val residualDF = arModel.transform(df)
      .withColumn(residual, -col(prediction) + col(label))
      .drop(features).drop(prediction)

    val newDF = TimeSeriesUtil.LagCombination(residualDF, residual, timeCol, q, lagsOnly = false)
      .filter(col(residual + lag + q).isNotNull)
    //      .drop(label)

    lr_ma.transform(newDF)
  }

  override def forecast(df: DataFrame, numAhead: Int): List[Double] = {
    require(q > 0, s"q can not be 0")

    if (lr_ma == null) fit(df)

    val newDF = transform(df)

    val prefix = "residual"
    val lag = "_lag_"
    val listDF = newDF.orderBy(desc(timeCol)).select(prefix + lag + 0)
      .limit(q).collect().map(_.getDouble(0)).toList

    var listPrediction = listDF
    listPrediction = tsFitDotProduct(listDF, q + 1, q, 0.0, getWeights(), meanValue = 0.0)

    var prediction = listPrediction.slice(0, q + 1).reverse
    prediction = prediction.map(i => i + getIntercept())

    ((q + 2) to numAhead) foreach { _ =>
      prediction = prediction :+ prediction(q)
    }
    prediction
  }


  def getIntercept(): Double = {
    lr_ma.getIntercept()
  }

  def getWeights(): Vector = {
    lr_ma.getWeights()
  }

  override def copy(): Model = {
    new MovingAverage(inputCol, timeCol, q,
      regParam, standardization, elasticNetParam, withIntercept, meanOut)
  }

  override def save(path: String): Unit = {
    MovingAverage.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    MovingAverage.saveHDFS(sc, this, path)
  }
}

object MovingAverage extends SaveLoad[MovingAverage] {
  def apply(uid: String, inputCol: String,
            timeCol: String, q: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean, meanOut: Boolean):
  MovingAverage = new MovingAverage(uid, inputCol, timeCol, q, regParam, standardization,
    elasticNetParam, withIntercept, meanOut)

  def apply(inputCol: String, timeCol: String, q: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean, meanOut: Boolean = false):
  MovingAverage = new MovingAverage(inputCol, timeCol, q, regParam, standardization,
    elasticNetParam, withIntercept, meanOut)
}
