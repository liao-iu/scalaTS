package com.suning.mlaas.spark.mllib.metric

import com.suning.mlaas.spark.mllib.util.Identifiable
import org.apache.spark.sql.DataFrame

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

abstract class Metrics(val uid: String, labelCol: String, predCol: String) extends Serializable{

  def this(labelCol: String, predCol: String) =
    this(Identifiable.randomUID("Metric"), labelCol, predCol)


  def getMetrics[T](df: DataFrame): Map[String, Double]
}
