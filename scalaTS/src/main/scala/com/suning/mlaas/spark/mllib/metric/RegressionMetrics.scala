package com.suning.mlaas.spark.mllib.metric

import org.apache.spark.ml.evaluation.RegressionEvaluator
import com.suning.mlaas.spark.mllib.util.Identifiable
import org.apache.spark.sql.DataFrame

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

class RegressionMetrics(override val uid: String, labelCol: String, predCol: String)
  extends Metrics(uid, labelCol, predCol){
  def this(labelCol: String, predCol: String) =
    this(Identifiable.randomUID("RegressionMetrics"), labelCol, predCol)

  def getMetrics[T](df: DataFrame): Map[String, Double] = {

    var metricMap = Map[String, Double]()
    val evaluator = new RegressionEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol(predCol)

    val mseValue = evaluator.setMetricName("mse").evaluate(df)
    val rmseValue = evaluator.setMetricName("rmse").evaluate(df)
    val r2Value = evaluator.setMetricName("r2").evaluate(df)
    val maeValue = evaluator.setMetricName("mae").evaluate(df)

    metricMap += ("mse" -> mseValue)
    metricMap += ("rmse" -> rmseValue)
    metricMap += ("r2" -> r2Value)
    metricMap += ("mae" -> maeValue)
    metricMap
  }
}


object RegressionMetrics {
  def apply(uid: String, labelCol: String, predCol: String):
  RegressionMetrics = new RegressionMetrics(uid, labelCol, predCol)

  def apply(labelCol: String, predCol: String):
  RegressionMetrics = new RegressionMetrics(labelCol, predCol)
}
