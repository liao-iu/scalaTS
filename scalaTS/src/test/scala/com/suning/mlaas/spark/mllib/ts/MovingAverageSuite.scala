package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */

class MovingAverageSuite extends SparkSuite {
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

  test("MovingAverage") {
    //    Not identical to R's implementation. Will try Yule-Walker.
    //    fit2<-arima(unemploy,order=c(0,0,5))
    //    fit2
    //    predict(fit2,n.ahead=10)$pred

    df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val maxP = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_ma = MovingAverage(inputCol, timeCol, maxP, regParam, standardization, elasticNetParam, withIntercept)
    // Fit the model
    val model1 = lr_ma.fit(df)
    val pred1 = model1.transform(df)

    pred1.show(10)

    val intercept = lr_ma.getIntercept()
    val weights = lr_ma.getWeights

    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    val numAhead = 10

    val predValues = TimeSeriesUtil.tsForecastMA(pred1, numAhead,
      inputCol, timeCol, maxP, intercept, weights)

    println(s"Prediction for ${numAhead} number ahead: ${predValues} ")

    //    or use forecast function

    val forecast = lr_ma.forecast(df, numAhead)
    forecast.foreach(println)

  }
}
