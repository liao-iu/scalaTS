package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite
import com.suning.mlaas.spark.mllib.ts.TimeSeriesUtil._
import org.apache.spark.sql.functions._

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class ARYuleWalkerSuite extends SparkSuite {
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

  test("AutoRegressive with YuleWalker") {
    //    Identical to R's implementation with ar() by yule-waler.
    //    datain<-read.csv("economics-timeseries.csv")
    //    unemploy<-datain$unemploy
    //    fit1<-ar(unemploy,order.max=5,method='yw',se.fit = TRUE)
    //    summary(fit1)
    //    fit1
    //    predict(fit1, n.ahead = 10)$pred
    val p = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val meanValue = getDouble(df.select(mean(inputCol)).collect()(0).get(0))

    val ar_yw = ARYuleWalker(inputCol, timeCol, p)

    // Fit the model
    val model1 = ar_yw.fit(df)

    val pred1 = model1.transform(df)
    val coefficients = ar_yw.getCoefficients()
    println(s"Coefficients: ${coefficients}")

    val weights = coefficients

    val numAhead = 10

    val predValues = TimeSeriesUtil.tsForecastYuleWalker(pred1, numAhead,
      inputCol, timeCol, p, weights, meanValue)

    println(s"Prediction for ${numAhead} number ahead: ${predValues} ")

    //    or use forecast function
    val forecast = ar_yw.forecast(df, numAhead)
    forecast.foreach(println)

    pred1.show()
    //    get the residuals as (-truth + predicted)
    val residuals = pred1.withColumn("residual", -col("prediction") + col("label")).select("residual")
    residuals.show(10)


    //      prediction metric has not been done yet
    //          val predResult = lr_ar.transform(df)
    //
    //      val predRDD = predResult.select("date", "unemploy").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    //      val regMetric = new RegressionMetrics(predRDD)
    //      val rmseSpark = regMetric.rootMeanSquaredError
  }
}
