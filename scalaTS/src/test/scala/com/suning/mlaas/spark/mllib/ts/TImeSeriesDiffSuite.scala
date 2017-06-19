package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

class TimeSeriesDiffSuite extends SparkSuite {
  override def beforeAll(): Unit = {
    super.beforeAll()
    //df = sparkSession.read
    // Window functions needs sparkSession

    df = sparkSession.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("data/economics-timeseries.csv")
  }

  test("Test time series Diff") {
    df.printSchema()

    val Diff = TimeSeriesDiff("unemploy", "date", 2, 3)
    val output = Diff.transform(df)
    //    output.withColumn("diff", col("unemploy")).show()
    output.show()
  }
}
