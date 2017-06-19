package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

class TimeSeriesSMASuite extends SparkSuite {
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

  test("Test time series lag SMA") {
    //    R's implementation
    //    require("TTR")
    //    datain<-read.csv("economics-timeseries.csv")
    //    unemploy<-datain$unemploy
    //    fit.sma<-SMA(unemploy, 7)
    //    fit.sma
    df.printSchema()
    val SMA = TimeSeriesSMA("unemploy", "date", 7)
    val output = SMA.transform(df)

    output.show(500)

    //    forecast with numAhead = 10
    val forecast = SMA.forecast(df, 10)

    forecast.foreach(println)
  }
}
