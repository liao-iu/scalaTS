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
  
class DiffAutoRegression(override val uid: String, inputCol: String, timeCol: String,
                         p: Int, d: Int,
                         regParam: Double, standardization: Boolean, elasticNetParam: Double,
                         withIntercept: Boolean)
  extends TSModel(uid, inputCol, timeCol) {

  def this(inputCol: String, timeCol: String, p: Int, d: Int, regParam: Double,
           standardization: Boolean = true, elasticNetParam: Double,
           withIntercept: Boolean = false) =
    this(Identifiable.randomUID("DiffAutoRegression"), inputCol, timeCol, p, d, regParam,
      standardization, elasticNetParam, withIntercept)

  private var lr_ar: LinearRegression = _

  //  private val lr_ar = new SparkLR().setRegParam(0.0)
  //    .setFitIntercept(false)
  //    .setStandardization(withIntercept)
  //    .setElasticNetParam(0.0)
  //    .setMaxIter(100)
  //    .setTol(1E-6)


  override def fitImpl(df: DataFrame): this.type = {

    require(p > 0 && d > 0, s"p and d can not be 0")

    val newDF = TimeSeriesUtil.DiffCombination(df, inputCol, timeCol, p, d)
    newDF.show(10)

    val r = 1 to p toArray

    val features = r.map(inputCol + "_diff_" + d + "_lag_" + _)

    val label = inputCol + "_diff_" + d + "_lag_0"


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
    require(p > 0 && d > 0, s"p and d can not be 0")

    val newDF = TimeSeriesUtil.DiffCombination(df, inputCol, timeCol, p, d, false)

    lr_ar.transform(newDF)
  }

  override def forecast(df: DataFrame, numAhead: Int): List[Double] = {
    require(p > 0 && d > 0, s"p and d can not be 0")

    if (lr_ar == null) fit(df)

    val newDF = transform(df).orderBy(desc(timeCol))

    val diff = "_diff_" + d
    val lag = "_lag_"
    var listDiff = newDF.select(inputCol + diff + lag + 0)
      .limit(p).collect().map(_.getDouble(0)).toList

    var listPrev = List[Double](
      getDouble(newDF.select(inputCol).limit(1).collect()(0).get(0))
    )

    val weights = getWeights()
    val intercept = getIntercept()

    (0 until numAhead).foreach {
      j => {
        val vec = Vectors.dense(listDiff.slice(0, p).toArray)
        var diff = 0.0
        (0 until p).foreach(
          i => {
            diff += vec(i) * weights(i)
          }
        )
        diff = diff + intercept

        listDiff = diff :: listDiff
        listPrev = (diff + listPrev(0)) :: listPrev
      }
    }
    listPrev.reverse.tail
  }


  def getIntercept(): Double = {
    lr_ar.getIntercept()
  }

  def getWeights(): Vector = {
    lr_ar.getWeights()
  }

  override def copy(): Model = {

    new DiffAutoRegression(inputCol, timeCol, p, d, regParam, standardization, elasticNetParam,
      withIntercept)
  }

  override def save(path: String): Unit = {
    DiffAutoRegression.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    DiffAutoRegression.saveHDFS(sc, this, path)
  }
}

object DiffAutoRegression extends SaveLoad[DiffAutoRegression] {
  def apply(uid: String, inputCol: String,
            timeCol: String, maxDiff: Int, diff: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean):
  DiffAutoRegression = new DiffAutoRegression(uid, inputCol, timeCol, maxDiff, diff, regParam,
    standardization, elasticNetParam, withIntercept)

  def apply(inputCol: String, timeCol: String, maxDiff: Int, diff: Int, regParam: Double,
            standardization: Boolean, elasticNetParam: Double, withIntercept: Boolean):
  DiffAutoRegression = new DiffAutoRegression(inputCol, timeCol, maxDiff, diff, regParam,
    standardization, elasticNetParam, withIntercept)
}
