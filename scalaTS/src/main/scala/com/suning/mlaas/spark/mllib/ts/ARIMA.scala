package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.regression.LinearRegression
import com.suning.mlaas.spark.mllib.ts.TimeSeriesUtil._
import com.suning.mlaas.spark.mllib.util.{Identifiable, Model, SaveLoad}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */

/** For the ARIMA(p,d,q) parameters,
  * p: order of the autoregressive part;
  * d: degree of first differencing involved;
  * q: order of the moving average part.
  */

class ARIMA(override val uid: String, inputCol: String, timeCol: String, p: Int, d: Int, q: Int,
            regParam: Double, standardization: Boolean, elasticNetParam: Double,
            withIntercept: Boolean, meanOut: Boolean)
  extends TSModel(uid, inputCol, timeCol) {


  def this(inputCol: String, timeCol: String, p: Int, d: Int, q: Int,
           regParam: Double, standardization: Boolean = true,
           elasticNetParam: Double, withIntercept: Boolean = false, meanOut: Boolean) =
    this(Identifiable.randomUID("ARIMA"), inputCol, timeCol, p, d, q, regParam, standardization,
      elasticNetParam, withIntercept, meanOut)

  private var lr_arima: LinearRegression = _
  private var armaModel: ARMA = _
  private var darModel: DiffAutoRegression = _


  def fitARIMA(df: DataFrame): Unit = {

    val lag = "_lag_"
    val label = "label"
    val prediction = "prediction"
    val residual = "residual"
    val feature = "features"

    // calculate residual
    val maxPQ = math.max(p, q)

    //For MA model.

    val arModel = AutoRegression(inputCol, timeCol, maxPQ,
      regParam, standardization, elasticNetParam, withIntercept, meanOut)

    arModel.fit(df)

    val pred_ma = arModel.transform(df)
    //    pred_ma.show(10)

    // get the residuals as (-truth + predicted)
    val residualDF = pred_ma.withColumn(residual, -col(prediction) + col(label))
    //    residualDF.show(10)

    var newDF_dee = TimeSeriesUtil.DiffCombination(residualDF, inputCol, timeCol, p, d, lagsOnly = false)
    //    newDF_dee.show(10)

    val newDF_arima = TimeSeriesUtil.LagCombinationARMA(newDF_dee,
      inputCol, residual, timeCol, prediction, p = p, q = q, d = d, arimaFeature = true)
      .filter(col(residual + lag + q).isNotNull)
      .drop(prediction)
      .drop(feature)
    //.withColumnRenamed(prediction, "predicitonResiduals")

    //    newDF_arima.show(10)

    val r = 1 to q
    val features_ma = r.map(residual + lag + _).toArray

    val diff = "_diff_" + d
    val r_ar = 1 to p
    val features_ar = r_ar.map(inputCol + diff + lag + _).toArray

    //    val features_ma = lagcol_ma.slice(1, count.toInt)

    val features = features_ar ++ features_ma

    //    features.foreach(println)



    val arimaLabel = inputCol + diff + "_lag_" + 0

    val maxIter = 1000
    val tol = 1E-6

    newDF_arima.persist()

    lr_arima = LinearRegression(features, arimaLabel, regParam, withIntercept, standardization, elasticNetParam, maxIter, tol)

    lr_arima.fit(newDF_arima)

    newDF_arima.unpersist()
  }

  override def fitImpl(df: DataFrame): this.type = {
    //    require(p > q, "For ARIMA(p,d,q), p should be large than q.")
    //Here is to introduce differencing, d for diff.
    require(p > 0 || q > 0, s"p or q can not be 0 at the same time")
    if (d == 0) {

      armaModel =
        ARMA(inputCol, timeCol, p, q, regParam, standardization, elasticNetParam, withIntercept)
      armaModel.fit(df)
    }
    else {
      if (q == 0) {
        darModel = DiffAutoRegression(inputCol, timeCol, p, d, regParam, standardization,
          elasticNetParam, withIntercept)
        darModel.fit(df)
      }
      else {
        fitARIMA(df)
      }
    }
    this
  }

  def transformARIMA(df: DataFrame): DataFrame = {
    val prefix = if (meanOut) "_meanOut" else ""
    val lag = "_lag_"
    val label = "label"
    val feature = "features"
    val prediction = "prediction"
    val residual = "residual"

    val maxPQ = math.max(p, q)

    val newDF = TimeSeriesUtil.LagCombination(df, inputCol,
      timeCol, maxPQ, lagsOnly = false, meanOut).filter(col(inputCol + prefix + lag + maxPQ).isNotNull)
    //For MA model.
    val lr_ar = AutoRegression(inputCol, timeCol, maxPQ,
      regParam, standardization, elasticNetParam, withIntercept, meanOut = false)

    lr_ar.fit(newDF)

    val pred_ma = lr_ar.transform(newDF)
    //    pred_ma.show()

    // get the residuals as (-truth + predicted)
    val residualDF = pred_ma.withColumn(residual, -col(prediction) + col(label))

    var newDF_dee = TimeSeriesUtil.DiffCombination(residualDF, inputCol, timeCol, p, d, false)

    val newDF_arima = TimeSeriesUtil.LagCombination(newDF_dee,
      residual, timeCol, q, lagsOnly = false, meanOut = false)
      .filter(col(residual + lag + q).isNotNull)
      .drop(prediction)
      .drop(label)
      .drop(feature)
    //        .withColumnRenamed("prediction", "predicitonResiduals")
    lr_arima.transform(newDF_arima)
  }

  override def transformImpl(df: DataFrame): DataFrame = {

    require(p > 0 || q > 0, s"p or q can not be 0 at the same time")
    if (d == 0) {
      armaModel.transform(df)
    }
    else {
      if (q == 0) {
        darModel.transform(df)
      }
      else {
        transformARIMA(df)
      }
    }
  }

  def forecastARIMA(df: DataFrame, numAhead: Int): List[Double] = {
    if (lr_arima == null) {
      fitARIMA(df)
    }
    val newDF = transformARIMA(df).orderBy(desc(timeCol))

    val diff = "_diff_" + d
    val residual = "residual"
    val lag = "_lag_"
    var listDiff = newDF.select(inputCol + diff + lag + 0)
      .limit(p).collect().map(_.getDouble(0)).toList
    var listResi = newDF.select(residual + lag + 0)
      .limit(q).collect().map(_.getDouble(0)).toList

    var listPrev = List[Double](
      getDouble(newDF.select(inputCol).limit(1).collect()(0).get(0))
    )

    val weights = getWeights()
    val intercept = getIntercept()

    val weightsDAR = Vectors.dense(weights.toArray.slice(0, p))
    val weightsMA = Vectors.dense(weights.toArray.slice(p, p + q))

    (0 until numAhead).foreach {
      j => {
        val vecDAR = Vectors.dense(listDiff.slice(0, p).toArray)
        val vecMA = Vectors.dense(listResi.slice(0, q).toArray)
        var diff = 0.0
        var resi = 0.0
        (0 until p).foreach(
          i => {
            diff += vecDAR(i) * weightsDAR(i)
          }
        )
        (0 until q).foreach(
          i => {
            resi += vecMA(i) * weightsMA(i)
          }
        )
        diff = diff + resi + intercept

        listDiff = diff :: listDiff
        listResi = resi :: listResi
        listPrev = (diff + listPrev(0)) :: listPrev
      }
    }
    listPrev.reverse.tail
  }

  override def forecast(df: DataFrame, numAhead: Int): List[Double] = {
    require(p > 0 || q > 0, s"p or q can not be 0 at the same time")

    if (armaModel == null || darModel == null || lr_arima == null) {
      fit(df)
    }

    if (d == 0) {
      armaModel.forecast(df, numAhead)
    }
    else {
      if (q == 0) {
        darModel.forecast(df, numAhead)
      }
      else {
        forecastARIMA(df, numAhead)
      }
    }
  }


  def getIntercept(): Double = {
    if (d == 0) {
      armaModel.getIntercept()
    }
    else {
      if (q == 0) {
        darModel.getIntercept()
      }
      else {
        lr_arima.getIntercept()
      }
    }
  }

  def getWeights(): Vector = {
    if (d == 0) {
      armaModel.getWeights()
    }
    else {
      if (q == 0) {
        darModel.getWeights()
      }
      else {
        lr_arima.getWeights()
      }
    }
  }

  override def copy(): Model = {
    new ARIMA(inputCol, timeCol, p, d, q, regParam, standardization, elasticNetParam, withIntercept, meanOut)
  }

  override def save(path: String): Unit = {
    ARIMA.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    ARIMA.saveHDFS(sc, this, path)
  }
}

object ARIMA extends SaveLoad[ARIMA] {
  def apply(uid: String, inputCol: String, timeCol: String, p: Int, d: Int, q: Int,
            regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean, meanOut: Boolean):
  ARIMA =
    new ARIMA(uid, inputCol, timeCol, p, d, q, regParam, standardization, elasticNetParam,
      withIntercept, meanOut)

  def apply(inputCol: String, timeCol: String, p: Int, d: Int, q: Int,
            regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean, meanOut: Boolean = true):
  ARIMA =
    new ARIMA(inputCol, timeCol, p, d, q, regParam, standardization, elasticNetParam, withIntercept, meanOut)

}
