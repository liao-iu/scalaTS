package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.regression.LinearRegression
import com.suning.mlaas.spark.mllib.util.{Identifiable, Model, SaveLoad}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class ARMA(override val uid: String, inputCol: String, timeCol: String, p: Int, q: Int,
           regParam: Double, standardization: Boolean, elasticNetParam: Double,
           withIntercept: Boolean)
  extends TSModel(uid, inputCol, timeCol) {
  /*
    For the ARMA(p,q) parameters,
   p: order of the autoregressive part;
   q: order of the moving average part.
   */

  def this(inputCol: String, timeCol: String, p: Int, q: Int, regParam: Double,
           standardization: Boolean = true,
           elasticNetParam: Double, withIntercept: Boolean = false) =
    this(Identifiable.randomUID("ARMA"), inputCol, timeCol, p, q, regParam, standardization,
      elasticNetParam, withIntercept)

  private var lr_arma: LinearRegression = _
  private var arModel: AutoRegression = _
  private var maModel: MovingAverage = _


  def fitARMA(df: DataFrame): Unit = {

    val lag = "_lag_"
    val residual = "residual"
    val label = "label"
    val prediction = "prediction"
    val feature = "features"
    val maxPQ = math.max(p, q)

    val newDF = TimeSeriesUtil.LagCombination(df, inputCol, timeCol, maxPQ).
      filter(col(inputCol + lag + maxPQ).isNotNull)

    //For MA model.
    val lr_ar = AutoRegression(inputCol, timeCol, maxPQ, regParam, standardization, elasticNetParam, false, false)

    val model_ma = lr_ar.fit(newDF)

    val pred_ma = model_ma.transform(newDF)

    // get the residuals as (-truth + predicted)
    val residualDF = pred_ma.withColumn(residual, -col(prediction) + col(label))

    val newDF_arma = TimeSeriesUtil.LagCombinationARMA(residualDF, inputCol, residual, timeCol,
      prediction, p = p, q = q)
      .filter(col(residual + lag + q).isNotNull)
      .drop(prediction)
      .drop(feature)

    val features_ar = (1 to p).map(inputCol + lag + _).toArray

    val features_ma = (1 to q).map(residual + lag + _).toArray


    val features = features_ar ++ features_ma

    val armaLabel = inputCol + lag + 0

    val maxIter = 1000
    val tol = 1E-6

    newDF_arma.persist()

    lr_arma = LinearRegression(features, armaLabel, regParam, withIntercept, standardization,
      elasticNetParam, maxIter, tol)

    lr_arma.fit(newDF_arma)

    newDF_arma.unpersist()
  }

  override def fitImpl(df: DataFrame): this.type = {
    require(p > 0 || q > 0, s"p or q can not be 0 at the same time")
    if (p == 0 || q == 0) {
      if (q == 0) {
        arModel = AutoRegression(inputCol, timeCol, p, regParam, standardization, elasticNetParam,
          withIntercept)
        arModel.fit(df)
      }
      else {
        maModel = MovingAverage(inputCol, timeCol, q, regParam, standardization, elasticNetParam,
          withIntercept)
        maModel.fit(df)
      }
    }
    else {
      fitARMA(df)
    }
    this
  }

  def transformARMA(df: DataFrame): DataFrame = {
    val lag = "_lag_"
    val residual = "residual"
    val label = "label"
    val prediction = "prediction"
    val feature = "features"

    val maxPQ = math.max(p, q)
    val newDF = TimeSeriesUtil.LagCombination(df, inputCol, timeCol, maxPQ).
      filter(col(inputCol + lag + maxPQ).isNotNull)
    //For MA model.
    val lr_ar = AutoRegression(inputCol, timeCol, maxPQ, regParam, standardization, elasticNetParam, withIntercept = false, meanOut = false)

    val model_ma = lr_ar.fit(newDF)
    val pred_ma = model_ma.transform(newDF)
    // get the residuals as (-truth + predicted)
    val residualDF = pred_ma.withColumn(residual, -col(prediction) + col(label))
    val newDF_arma = TimeSeriesUtil.LagCombinationARMA(residualDF, inputCol, residual, timeCol,
      prediction, p = p, q = q)
      .filter(col(residual + lag + q).isNotNull)
      .drop(prediction)

    lr_arma.transform(newDF_arma)
  }

  override def forecast(df: DataFrame, numAhead: Int): List[Double] = {

    require(p > 0 || q > 0, s"p or q can not be 0 at the same time")

    if (lr_arma == null || arModel == null || maModel == null) fit(df)

    val newDF = transform(df)

    if (p == 0 || q == 0) {
      if (q == 0) {
        TimeSeriesUtil.tsForecastAR(newDF, numAhead, inputCol, timeCol, p,
          getIntercept(), getWeights())
      }
      else {
        TimeSeriesUtil.tsForecastMA(newDF, numAhead, inputCol, timeCol, q,
          getIntercept(), getWeights())
      }
    }
    else {
      TimeSeriesUtil.tsForecastARMAModel(newDF, numAhead, inputCol, timeCol, p, q,
        getIntercept(), getWeights())
    }
  }

  override def transformImpl(df: DataFrame): DataFrame = {
    require(p > 0 || q > 0, s"p or q can not be 0 at the same time")
    if (p == 0 || q == 0) {
      if (q == 0) {
        arModel.transform(df)
      }
      else {
        maModel.transform(df)
      }
    }
    else {
      transformARMA(df)
    }
  }


  def getIntercept(): Double = {
    if (p == 0 || q == 0) {
      if (q == 0) {
        arModel.getIntercept()
      }
      else {
        maModel.getIntercept()
      }
    }
    else {
      lr_arma.getIntercept()
    }
  }

  def getWeights(): Vector = {
    if (p == 0 || q == 0) {
      if (q == 0) {
        arModel.getWeights()
      }
      else {
        maModel.getWeights()
      }
    }
    else {
      lr_arma.getWeights()
    }
  }

  override def copy(): Model = {
    new ARMA(inputCol, timeCol, p, q, regParam, standardization, elasticNetParam, withIntercept)
  }

  override def save(path: String): Unit = {
    ARMA.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    ARMA.saveHDFS(sc, this, path)
  }
}

object ARMA extends SaveLoad[ARMA] {
  def apply(uid: String, inputCol: String,
            timeCol: String, p: Int, q: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean):
  ARMA =
    new ARMA(uid, inputCol, timeCol, p, q, regParam, standardization, elasticNetParam,
      withIntercept)

  def apply(inputCol: String, timeCol: String, p: Int, q: Int, regParam: Double,
            standardization: Boolean, elasticNetParam: Double, withIntercept: Boolean):
  ARMA = new ARMA(inputCol, timeCol, p, q, regParam, standardization, elasticNetParam, withIntercept)

}
