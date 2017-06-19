package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite
import org.apache.spark.sql.functions._

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

class TimeSeriesLagSuite extends SparkSuite {
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

  test("Test time series lag") {
    df.printSchema()

    val lag = TimeSeriesLag("unemploy", "date", 7)
    val output = lag.transform(df)
    output.withColumn("diff", col("unemploy") - 1000).show()
    //    output.show()
  }
}
