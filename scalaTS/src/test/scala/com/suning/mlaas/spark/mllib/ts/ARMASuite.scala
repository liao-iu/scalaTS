package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */

class ARMASuite extends SparkSuite {
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

  test("AutoRegressiveIntegratedMovingAverageSuite p=0") {
    df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p = 0
    val q = 1
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_arma = ARMA(inputCol, timeCol, p, q, regParam, standardization, elasticNetParam, withIntercept)
    // Fit the model
    lr_arma.fit(df)
    val pred1 = lr_arma.transform(df)

    val intercept = lr_arma.getIntercept()
    val weights = lr_arma.getWeights

    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    val numAhead = 10

    val predValues = TimeSeriesUtil.tsForecastARMA(pred1, numAhead,
      inputCol, timeCol, p, q, intercept, weights)

    println(s"Prediction for ${numAhead} number ahead: ${predValues} ")

    //    or use forecast function

    val forecast = lr_arma.forecast(df, numAhead)
    forecast.foreach(println)
  }

  test("AutoRegressiveIntegratedMovingAverageSuite q=0") {
    df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p = 3
    val q = 0
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_arma = ARMA(inputCol, timeCol, p, q, regParam, standardization, elasticNetParam, withIntercept)
    // Fit the model
    lr_arma.fit(df)
    val pred1 = lr_arma.transform(df)

    val intercept = lr_arma.getIntercept()
    val weights = lr_arma.getWeights

    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    val numAhead = 10

    val predValues = TimeSeriesUtil.tsForecastARMA(pred1, numAhead,
      inputCol, timeCol, p, q, intercept, weights)

    println(s"Prediction for ${numAhead} number ahead: ${predValues} ")

    //    or use forecast function

    val forecast = lr_arma.forecast(df, numAhead)
    forecast.foreach(println)
  }

  test("AutoRegressiveIntegratedMovingAverageSuite") {
    df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p = 3
    val q = 2
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_arma = ARMA(inputCol, timeCol, p, q, regParam, standardization, elasticNetParam, withIntercept)
    // Fit the model
    lr_arma.fit(df)
    val pred1 = lr_arma.transform(df)
    pred1.show(10)

    val intercept = lr_arma.getIntercept()
    val weights = lr_arma.getWeights

    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    val numAhead = 10

    val predValues = TimeSeriesUtil.tsForecastARMA(pred1, numAhead,
      inputCol, timeCol, p, q, intercept, weights)

    println(s"Prediction for ${numAhead} number ahead: ${predValues} ")

    //    or use forecast function

    val forecast = lr_arma.forecast(df, numAhead)
    forecast.foreach(println)
  }

}
