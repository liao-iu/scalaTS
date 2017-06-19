package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

class TimeSeriesDecomposeSuite extends SparkSuite {

  override def beforeAll(): Unit = {
    super.beforeAll()

    df = sparkSession.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("data/economics-timeseries.csv")
  }


  test("test the time series decompose addtive") {
    println("test the time series decompose addtive")

    println(df.count)

    val colName = "pce"

    val dateCol = "date"

    // default model is addtive
    // default period is 12

    val obj = new TimeSeriesDecompose(colName, dateCol, model = "additive")
    val result = obj.transform(df)
    result.show(20)
    println(result.count)


  }

  test("test the time series decompose multiplicative default") {
    println("test the time series decompose addtive")

    println(df.count)

    val colName = "pce"
    val dateCol = "date"


    // default model is addtive
    // default period is 12

    val obj = new TimeSeriesDecompose(colName, dateCol, period = 12, model = "multiplicative")
    val result = obj.transform(df)
    result.show(20)
    println(result.count)


  }


  test("test the time series decompose addtive change the period") {
    println("test the time series decompose addtive")
    val colName = "pce"

    val dateCol = "date"


    val obj = new TimeSeriesDecompose(colName, dateCol, period = 100, model = "additive")

    val result = obj.transform(df)

    result.show(200)

  }


  test("test the time series decompose multiplicative") {
    println("test the time series decompose multiplicative")
    val colName = "pce"

    val dateCol = "date"


    val obj = new TimeSeriesDecompose(colName, dateCol, period = 10, model = "multiplicative")

    val result = obj.transform(df)

    result.show(20)

  }


}
