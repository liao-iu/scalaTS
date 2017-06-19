package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.util.{Identifiable, Model, SaveLoad}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import collection.mutable.Map

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class AutoARMA(override val uid: String, inputCol: String, timeCol: String, p_Max: Int, q_Max: Int,
               regParam: Double, standardization: Boolean, elasticNetParam: Double,
               withIntercept: Boolean, criterion: String)

  extends TSModel(uid, inputCol, timeCol) {
  /*
    For the ARMA(p,q) parameters,
   p: order of the autoregressive part;
   q: order of the moving average part.
   */

  def this(inputCol: String, timeCol: String, p_Max: Int, q_Max: Int,
           regParam: Double, standardization: Boolean = true, elasticNetParam: Double,
           withIntercept: Boolean = false, criterion: String) =
    this(Identifiable.randomUID("AutoARMA"), inputCol, timeCol, p_Max, q_Max, regParam, standardization,
      elasticNetParam, withIntercept, criterion)

  private var lr_Autoarma: ARMA = _
  var criterionValue = Map[(Int, Int), Double]()
  private var p_Best: Int = _
  private var q_Best: Int = _
  var best: ((Int, Int), Double) = _


  override def fitImpl(df: DataFrame): this.type = {
    val n = df.count().toInt
    criterionValue = criterionCalcul(df, n, criterion)

    println(criterionValue)

    p_Best = (criterionValue.minBy(_._2)._1)._1

    q_Best = (criterionValue.minBy(_._2)._1)._2

    println(s"Best criterion value is ${criterionValue.valuesIterator.min} by p: ${p_Best} and q: ${q_Best}")
    //
    df.persist()
    lr_Autoarma = ARMA(inputCol, timeCol, p_Best, q_Best,
      regParam, standardization, elasticNetParam, withIntercept)

    lr_Autoarma.fit(df)
    df.unpersist()

    this
  }


  def criterionCalcul(df: DataFrame, n: Int, criterion: String): Map[(Int, Int), Double] = {

    (1 to q_Max).map(i => {
      val lr_Autoarma = ARMA(inputCol, timeCol, 0, i,
        regParam, standardization, elasticNetParam, withIntercept)
      val model = lr_Autoarma.fit(df)
      val pred = model.transform(df)
      val residuals = pred.withColumn("residual", -col("prediction") + col("label")).select("residual")

      var criterionIte = TimeSeriesUtil.AIC(residuals, 1, n)

      if (criterion == "aic") {
        criterionIte = TimeSeriesUtil.AIC(residuals, i, n)
        println(s"AIC value for p: 0 and q: ${i} is ${criterionIte}")
      } else if (criterion == "bic") {
        criterionIte = TimeSeriesUtil.BIC(residuals, i, n)
        println(s"BIC value for p: 0 and q: ${i} is ${criterionIte}")
      } else {
        criterionIte = TimeSeriesUtil.AICc(residuals, i, n)
        println(s"AICC value for p: 0 and q: ${i} is ${criterionIte}")
      }
      criterionValue += (0, i) -> criterionIte
    })

    (1 to p_Max).map(i => {
      (0 to q_Max).map(j => {
        val lr_Autoarma = ARMA(inputCol, timeCol, i, j,
          regParam, standardization, elasticNetParam, withIntercept)
        val model = lr_Autoarma.fit(df)
        val pred = model.transform(df)
        val residuals = pred.withColumn("residual", -col("prediction") + col("label")).select("residual")

        var criterionIte = TimeSeriesUtil.AIC(residuals, 1, n)

        if (criterion == "aic") {
          criterionIte = TimeSeriesUtil.AIC(residuals, (i + j) , n)
          println(s"AIC value for p: ${i} and q: ${j} is ${criterionIte}")
        } else if (criterion == "bic") {
          criterionIte = TimeSeriesUtil.BIC(residuals, (i + j), n)
          println(s"BIC value for p: ${i} and q: ${j} is ${criterionIte}")
        } else {
          criterionIte = TimeSeriesUtil.AICc(residuals, (i + j), n)
          println(s"AICC value for p: ${i} and q: ${j} is ${criterionIte}")
        }
        criterionValue += (i, j) -> criterionIte
      })
    })

    criterionValue

  }

  // earlyStop to do

  //  def criterionCalcul(df: DataFrame, n: Int, criterion: String,
  //                      earlyStop: Boolean): Array[Double] = {
  //    var criterionValues = List(0.0)
  //
  //    val lr_ar = AutoRegression(inputCol, timeCol, 1,
  //      regParam, standardization, elasticNetParam, withIntercept)
  //    val model = lr_ar.fit(df)
  //    val pred = model.transform(df)
  //
  //    val residuals = pred.withColumn("residual", -col("prediction") + col("label")).select("residual")
  //
  //    breakable {
  //      if (criterion == "aic") {
  //        criterionValues = List(TimeSeriesUtil.AIC(residuals, 1, n))
  //        println(s"AIC value for Lag ${1} is ${criterionValues(0)}")
  //        for (i <- 2 to p_Max) {
  //          val lr_ar = AutoRegression(inputCol, timeCol, i,
  //            regParam, standardization, elasticNetParam, withIntercept, meanOut)
  //          val model = lr_ar.fit(df)
  //          val pred = model.transform(df)
  //          val residuals = pred.withColumn("residual", -col("prediction") + col("label")).select("residual")
  //
  //          criterionValues = criterionValues :+ TimeSeriesUtil.AIC(residuals, i, n)
  //
  //          println(s"AIC value for Lag ${i} is ${criterionValues(i - 1)}")
  //          if (criterionValues(i - 1) > criterionValues(i - 2)) break
  //
  //        }
  //      }
  //      else if (criterion == "bic") {
  //        criterionValues = List(TimeSeriesUtil.BIC(residuals, 1, n))
  //        println(s"BIC value for Lag ${1} is ${criterionValues(0)}")
  //        for (i <- 2 to p_Max) {
  //          val lr_ar = AutoRegression(inputCol, timeCol, i,
  //            regParam, standardization, elasticNetParam, withIntercept, meanOut)
  //          val model = lr_ar.fit(df)
  //          val pred = model.transform(df)
  //          val residuals = pred.withColumn("residual", -col("prediction") + col("label")).select("residual")
  //
  //          criterionValues = criterionValues :+ TimeSeriesUtil.BIC(residuals, i, n)
  //
  //          println(s"BIC value for Lag ${i} is ${criterionValues(i - 1)}")
  //
  //          if (criterionValues(i - 1) > criterionValues(i - 2)) break
  //
  //        }
  //      } else {
  //        criterionValues = List(TimeSeriesUtil.AICc(residuals, 1, n))
  //        println(s"AICC value for Lag ${1} is ${criterionValues(0)}")
  //
  //        for (i <- 2 to p_Max) {
  //          val lr_ar = AutoRegression(inputCol, timeCol, i,
  //            regParam, standardization, elasticNetParam, withIntercept, meanOut)
  //          val model = lr_ar.fit(df)
  //          val pred = model.transform(df)
  //          val residuals = pred.withColumn("residual", -col("prediction") + col("label")).select("residual")
  //
  //          criterionValues = criterionValues :+ TimeSeriesUtil.AICc(residuals, i, n)
  //
  //          println(s"AICC value for Lag ${i} is ${criterionValues(i - 1)}")
  //
  //          if (criterionValues(i - 1) > criterionValues(i - 2)) break
  //        }
  //      }
  //    }
  //    criterionValues.toArray
  //  }

  override def transformImpl(df: DataFrame): DataFrame = {
    lr_Autoarma.transform(df)
  }

  override def forecast(df: DataFrame, numAhead: Int): List[Double] = {
    lr_Autoarma.forecast(df, numAhead)
  }

  def getIntercept(): Double = {
    lr_Autoarma.getIntercept()
  }

  def getWeights(): Vector = {
    lr_Autoarma.getWeights
  }

  override def copy(): Model = {
    new AutoARMA(inputCol, timeCol, p_Max, q_Max, regParam, standardization, elasticNetParam,
      withIntercept, criterion)
  }

  override def save(path: String): Unit = {
    AutoARMA.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    AutoARMA.saveHDFS(sc, this, path)
  }
}

object AutoARMA extends SaveLoad[AutoARMA] {
  def apply(uid: String, inputCol: String,
            timeCol: String, p_Max: Int, q_Max: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean, criterion: String = "aic"):
  AutoARMA = new AutoARMA(uid, inputCol, timeCol, p_Max, q_Max, regParam, standardization,
    elasticNetParam, withIntercept, criterion)

  def apply(inputCol: String, timeCol: String, p_Max: Int, q_Max: Int, regParam: Double, standardization: Boolean,
            elasticNetParam: Double, withIntercept: Boolean, criterion: String):
  AutoARMA = new AutoARMA(inputCol, timeCol, p_Max, q_Max, regParam, standardization,
    elasticNetParam, withIntercept, criterion)
}
