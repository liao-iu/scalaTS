package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class DiffAutoRegressionSuite extends SparkSuite {
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

  test("DiffAutoRegression") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val maxLag = 5
    val diff = 1
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_ar = DiffAutoRegression(inputCol, timeCol, maxLag, diff, regParam, standardization, elasticNetParam, withIntercept)

    // Fit the model
    val model1 = lr_ar.fit(df)
    //
    val pred1 = model1.transform(df).show()
    //
    val intercept = lr_ar.getIntercept()
    val weights = lr_ar.getWeights
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    //      prediction metric has not been done yet
    //      val predResult = lr_ar.transform(df)
    //
    //      val predRDD = predResult.select("date", "unemploy").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    //      val regMetric = new RegressionMetrics(predRDD)
    //      val rmseSpark = regMetric.rootMeanSquaredError
  }

  test("DiffAutoRegression with diff 2") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val maxLag = 5
    val diff = 2
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_ar = DiffAutoRegression(inputCol, timeCol, maxLag, diff, regParam, standardization, elasticNetParam, withIntercept)

    // Fit the model
    val model1 = lr_ar.fit(df)
    //
    val pred1 = model1.transform(df).show()
    //
    val intercept = lr_ar.getIntercept()
    val weights = lr_ar.getWeights
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    //      prediction metric has not been done yet
    //      val predResult = lr_ar.transform(df)
    //
    //      val predRDD = predResult.select("date", "unemploy").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    //      val regMetric = new RegressionMetrics(predRDD)
    //      val rmseSpark = regMetric.rootMeanSquaredError
  }
  test("DiffAutoRegression forecast") {
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val maxLag = 5
    val diff = 2
    val inputCol = "unemploy"
    val timeCol = "date"
    val numAhead = 10

    val lr_ar = DiffAutoRegression(inputCol, timeCol, maxLag, diff, regParam, standardization, elasticNetParam, withIntercept)

    // Fit the model
    val forecast = lr_ar.forecast(df, numAhead)
    forecast.foreach(println)
    //

  }


}
