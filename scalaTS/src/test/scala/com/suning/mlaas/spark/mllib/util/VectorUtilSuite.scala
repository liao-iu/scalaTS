package com.suning.mlaas.spark.mllib.util

import com.suning.mlaas.spark.mllib.SparkSuite
import com.suning.mlaas.spark.mllib.transform.Feature2Vector
import com.suning.mlaas.spark.mllib.util.VectorUtil._

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

class VectorUtilSuite extends SparkSuite{

  override def beforeAll(): Unit = {
    super.beforeAll()
    df = sparkSession.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("data/iris.csv")
  }

  test("vector2String and string2Vector") {
    val features = Array("Petal_Length", "Sepal_Width", "Petal_Width")
    val label = "Sepal_Length"
    val f2v = Feature2Vector(features, label)
    val output = f2v.transform(df)

    val v2sOutput = vector2String(output)
    v2sOutput.show(5)
    v2sOutput.printSchema()
    val s2vOutput = string2Vector(v2sOutput)
    s2vOutput.show(5)
    s2vOutput.printSchema()

  }

}
