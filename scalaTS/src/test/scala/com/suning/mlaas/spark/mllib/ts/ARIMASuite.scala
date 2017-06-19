package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */

class ARIMASuite extends SparkSuite {
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

  //  def time[A](a: => A) = {
  //       val now = System.nanoTime
  //    val result = a
  //      val micros = (System.nanoTime - now) / 1000
  //      println("%d microseconds".format(micros))
  //      result
  //      }

  test("AutoRegressiveIntegratedMovingAverageSuite1 ar < ma") {
    df.printSchema()
    //    time {
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p = 200
    //val p = 3
    val d = 0
    val q = 0
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_arima = ARIMA(inputCol, timeCol, p, d, q, regParam, standardization, elasticNetParam, withIntercept)
    // Fit the model
    val model1 = lr_arima.fit(df)
    val pred1 = model1.transform(df) //.show()
    //    // get the residuals as (-truth + predicted)
    //    val residuals = pred1.withColumn("residual", - col("prediction") + col("label")).select("residual")
    //    residuals.show(10)

    val intercept = lr_arima.getIntercept()
    val weights = lr_arima.getWeights

    //    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    //      prediction metric has not been done yet
    //          val predResult = lr_ar.transform(df)
    //
    //      val predRDD = predResult.select("date", "unemploy").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    //      val regMetric = new RegressionMetrics(predRDD)
    //      val rmseSpark = regMetric.rootMeanSquaredError
    //  }
  }

  test("AutoRegressiveIntegratedMovingAverageSuite with Differnecing 1 or 2") {
    df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val meanOut = true
    val p = 2
    val d = 1
    val q = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_arima = ARIMA(inputCol, timeCol, p, d, q, regParam, standardization, elasticNetParam, withIntercept)
    // Fit the model
    val model1 = lr_arima.fit(df)
    val pred1 = model1.transform(df).show()
    //    //    pred1.show()
    //    // get the residuals as (-truth + predicted)
    //    val residuals = pred1.withColumn("residual", - col("prediction") + col("label")).select("residual")
    //    residuals.show(10)

    val intercept = lr_arima.getIntercept()
    val weights = lr_arima.getWeights

    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    //      prediction metric has not been done yet
    //          val predResult = lr_ar.transform(df)
    //
    //      val predRDD = predResult.select("date", "unemploy").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    //      val regMetric = new RegressionMetrics(predRDD)
    //      val rmseSpark = regMetric.rootMeanSquaredError
  }

  test("ARIMA forecast (2,1,5)") {
    df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val meanOut = true
    val p = 2
    val d = 1
    val q = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_arima = ARIMA(inputCol, timeCol, p, d, q, regParam, standardization, elasticNetParam, withIntercept)

    val forecast = lr_arima.forecast(df, 10)
    forecast.foreach(println)

  }

  test("ARIMA forecast (2,2,5)") {
    df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val meanOut = true
    val p = 2
    val d = 2
    val q = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_arima = ARIMA(inputCol, timeCol, p, d, q, regParam, standardization, elasticNetParam, withIntercept)

    val forecast = lr_arima.forecast(df, 10)
    forecast.foreach(println)

  }

  test("ARIMA forecast (2,0,5)") {
    df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val meanOut = true
    val p = 2
    val d = 0
    val q = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_arima = ARIMA(inputCol, timeCol, p, d, q, regParam, standardization, elasticNetParam, withIntercept)

    val forecast = lr_arima.forecast(df, 10)
    forecast.foreach(println)

  }

  test("ARIMA forecast (2,0,0)") {
    df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val meanOut = true
    val p = 2
    val d = 0
    val q = 0
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_arima = ARIMA(inputCol, timeCol, p, d, q, regParam, standardization, elasticNetParam, withIntercept)

    val forecast = lr_arima.forecast(df, 10)
    forecast.foreach(println)

  }
}
