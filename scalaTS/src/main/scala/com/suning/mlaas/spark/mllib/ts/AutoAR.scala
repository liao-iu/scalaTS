package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.util.{Identifiable, Model, SaveLoad}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import scala.util.control.Breaks._

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class AutoAR(override val uid: String, inputCol: String, timeCol: String, p_Max: Int,
             regParam: Double, standardization: Boolean, elasticNetParam: Double,
             withIntercept: Boolean, meanOut: Boolean, criterion: String, earlyStop: Boolean)
  extends TSModel(uid, inputCol, timeCol) {

  def this(inputCol: String, timeCol: String, p_Max: Int,
           regParam: Double, standardization: Boolean, elasticNetParam: Double,
           withIntercept: Boolean, meanOut: Boolean, criterion: String, earlyStop: Boolean) =
    this(Identifiable.randomUID("AutoAR"), inputCol, timeCol, p_Max, regParam, standardization,
      elasticNetParam, withIntercept, meanOut, criterion, earlyStop)

  def this(inputCol: String, timeCol: String, p_Max: Int, regParam: Double, standardization: Boolean,
           elasticNetParam: Double, withIntercept: Boolean,
           criterion: String, earlyStop: Boolean) =
    this(Identifiable.randomUID("AutoAR"), inputCol, timeCol, p_Max,
      regParam, standardization, elasticNetParam, withIntercept, false, criterion, earlyStop)


  private var lr_Autoar: AutoRegression = _
  private var criterionValue: Array[Double] = _
  private var p_Best: Int = _


  override def fitImpl(df: DataFrame): this.type = {
    val n = df.count().toInt

    if (earlyStop) {
      criterionValue = criterionCalcul(df, n, criterion, earlyStop = true)
    } else {
      criterionValue = criterionCalcul(df, n, criterion)
      //      aic.foreach(println)
    }

    p_Best = criterionValue.indexOf(criterionValue.min) + 1

    println(s"Best criterion value is ${criterionValue.min} by Lag ${p_Best}")

    df.persist()
    lr_Autoar = AutoRegression(inputCol, timeCol, p_Best,
      regParam, standardization, elasticNetParam, withIntercept, meanOut)

    lr_Autoar.fit(df)
    df.unpersist()

    this
  }


  def criterionCalcul(df: DataFrame, n: Int, criterion: String): Array[Double] = {
    criterionValue = (1 to p_Max toArray).map(i => {
      val lr_ar = AutoRegression(inputCol, timeCol, i,
        regParam, standardization, elasticNetParam, withIntercept, meanOut)
      val model = lr_ar.fit(df)
      val pred = model.transform(df)
      val residuals = pred.withColumn("residual", -col("prediction") + col("label")).select("residual")

      var criterionIte = TimeSeriesUtil.AIC(residuals, 1, n)
      if (criterion == "aic") {
        criterionIte = TimeSeriesUtil.AIC(residuals, i, n)
        println(s"AIC value for Lag ${i} is ${criterionIte}")
      } else if (criterion == "bic") {
        criterionIte = TimeSeriesUtil.BIC(residuals, i, n)
        println(s"BIC value for Lag ${i} is ${criterionIte}")
      } else {
        criterionIte = TimeSeriesUtil.AICc(residuals, i, n)
        println(s"AICC value for Lag ${i} is ${criterionIte}")
      }
      criterionIte
    })
    criterionValue
  }

  def criterionCalcul(df: DataFrame, n: Int, criterion: String,
                      earlyStop: Boolean): Array[Double] = {
    var criterionValues = List(0.0)

    val lr_ar = AutoRegression(inputCol, timeCol, 1,
      regParam, standardization, elasticNetParam, withIntercept, meanOut)
    val model = lr_ar.fit(df)
    val pred = model.transform(df)

    val residuals = pred.withColumn("residual", -col("prediction") + col("label")).select("residual")

    breakable {
      if (criterion == "aic") {
        criterionValues = List(TimeSeriesUtil.AIC(residuals, 1, n))
        println(s"AIC value for Lag ${1} is ${criterionValues(0)}")
        for (i <- 2 to p_Max) {
          val lr_ar = AutoRegression(inputCol, timeCol, i,
            regParam, standardization, elasticNetParam, withIntercept, meanOut)
          val model = lr_ar.fit(df)
          val pred = model.transform(df)
          val residuals = pred.withColumn("residual", -col("prediction") + col("label")).select("residual")

          criterionValues = criterionValues :+ TimeSeriesUtil.AIC(residuals, i, n)

          println(s"AIC value for Lag ${i} is ${criterionValues(i - 1)}")
          if (criterionValues(i - 1) > criterionValues(i - 2)) break

        }
      }
      else if (criterion == "bic") {
        criterionValues = List(TimeSeriesUtil.BIC(residuals, 1, n))
        println(s"BIC value for Lag ${1} is ${criterionValues(0)}")
        for (i <- 2 to p_Max) {
          val lr_ar = AutoRegression(inputCol, timeCol, i,
            regParam, standardization, elasticNetParam, withIntercept, meanOut)
          val model = lr_ar.fit(df)
          val pred = model.transform(df)
          val residuals = pred.withColumn("residual", -col("prediction") + col("label")).select("residual")

          criterionValues = criterionValues :+ TimeSeriesUtil.BIC(residuals, i, n)

          println(s"BIC value for Lag ${i} is ${criterionValues(i - 1)}")

          if (criterionValues(i - 1) > criterionValues(i - 2)) break

        }
      } else {
        criterionValues = List(TimeSeriesUtil.AICc(residuals, 1, n))
        println(s"AICC value for Lag ${1} is ${criterionValues(0)}")

        for (i <- 2 to p_Max) {
          val lr_ar = AutoRegression(inputCol, timeCol, i,
            regParam, standardization, elasticNetParam, withIntercept, meanOut)
          val model = lr_ar.fit(df)
          val pred = model.transform(df)
          val residuals = pred.withColumn("residual", -col("prediction") + col("label")).select("residual")

          criterionValues = criterionValues :+ TimeSeriesUtil.AICc(residuals, i, n)

          println(s"AICC value for Lag ${i} is ${criterionValues(i - 1)}")

          if (criterionValues(i - 1) > criterionValues(i - 2)) break
        }
      }
    }
    criterionValues.toArray
  }

  override def transformImpl(df: DataFrame): DataFrame = {
    lr_Autoar.transform(df)
  }

  override def forecast(df: DataFrame, numAhead: Int): List[Double] = {
    lr_Autoar.forecast(df, numAhead)
  }

  def getIntercept(): Double = {
    lr_Autoar.getIntercept()
  }

  def getWeights(): Vector = {
    lr_Autoar.getWeights
  }

  override def copy(): Model = {
    new AutoAR(inputCol, timeCol, p_Max, regParam, standardization, elasticNetParam,
      withIntercept, meanOut, criterion, earlyStop)
  }

  override def save(path: String): Unit = {
    AutoAR.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    AutoAR.saveHDFS(sc, this, path)
  }
}

object AutoAR extends SaveLoad[AutoAR] {
  def apply(uid: String, inputCol: String,
            timeCol: String, p_Max: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean, meanOut: Boolean = false,
            criterion: String = "aic", earlyStop: Boolean = false):
  AutoAR = new AutoAR(uid, inputCol, timeCol, p_Max, regParam, standardization,
    elasticNetParam, withIntercept, meanOut, criterion, earlyStop)

  def apply(inputCol: String, timeCol: String, p_Max: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean, meanOut: Boolean,
            criterion: String, earlyStop: Boolean):
  AutoAR = new AutoAR(inputCol, timeCol, p_Max, regParam, standardization,
    elasticNetParam, withIntercept, meanOut, criterion, earlyStop)

  def apply(inputCol: String, timeCol: String, p_Max: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean,
            criterion: String, earlyStop: Boolean):
  AutoAR = new AutoAR(inputCol, timeCol, p_Max, regParam, standardization,
    elasticNetParam, withIntercept, false, criterion, earlyStop)

}
