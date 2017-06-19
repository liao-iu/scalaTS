package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite
import com.suning.mlaas.spark.mllib.metric.RegressionMetrics
import com.suning.mlaas.spark.mllib.ts.TimeSeriesUtil._
import org.apache.spark.sql.functions._

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class AutoRegressionSuite extends SparkSuite {
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

  test("AutoRegression") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val maxLag = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val lr_ar = AutoRegression(inputCol, timeCol, maxLag,
      regParam, standardization, elasticNetParam, withIntercept)

    // Fit the model
    val model1 = lr_ar.fit(df)

    val pred1 = model1.transform(df)

    val intercept = lr_ar.getIntercept()
    val weights = lr_ar.getWeights

    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    val numHead = 10

    val predValues = TimeSeriesUtil.tsForecastAR(pred1, numHead,
      inputCol, timeCol, p = maxLag, intercept, weights)

    println(s"Prediction for ${numHead} number ahead: ${predValues} ")

    pred1.show()
    // get the residuals as (-truth + predicted)
    val residuals = pred1.withColumn("residual", -col("prediction") + col("label")).select("residual")
    residuals.show(10)
  }

  test("AutoRegression by taking mean out (default)") {
    //    Identical to R's implementation with ar() by ols.
    //    datain<-read.csv("economics-timeseries.csv")
    //    unemploy<-datain$unemploy
    //    fit1<-ar(unemploy,order.max=5,method='ols',se.fit = TRUE)
    //    summary(fit1)
    //    fit1
    //    predict(fit1, n.ahead = 10)$pred

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val maxLag = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val meanOut = true
    val meanValue = getDouble(df.select(mean(inputCol)).collect()(0).get(0))

    val lr_ar = AutoRegression(inputCol, timeCol, maxLag,
      regParam, standardization, elasticNetParam, withIntercept, meanOut)

    // Fit the model
    val model1 = lr_ar.fit(df)

    val pred1 = model1.transform(df)

    val intercept = lr_ar.getIntercept()
    val weights = lr_ar.getWeights

    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    val numAhead = 10

    val predValues = TimeSeriesUtil.tsForecastAR(pred1, numAhead,
      inputCol, timeCol, p = maxLag, intercept, weights, meanOut, meanValue)

    println(s"Prediction for ${numAhead} number ahead: ${predValues} ")

    //    or use forecast function

    val forecast = lr_ar.forecast(df, numAhead)
    forecast.foreach(println)

    pred1.show()
    // get the residuals as (-truth + predicted)
    val residuals = pred1.withColumn("residual", -col("prediction") + col("label")).select("residual")
    residuals.show(10)

  }

  test("AutoRegression forecast by taking mean out (default)") {
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val maxLag = 1
    val inputCol = "unemploy"
    val timeCol = "date"
    val meanOut = true


    val lr_ar = AutoRegression(inputCol, timeCol, maxLag,
      regParam, standardization, elasticNetParam, withIntercept, meanOut)

    val numHead = 10

    val forecast = lr_ar.forecast(df, numHead)
    forecast.foreach(println)
  }

  test("AutoRegression AIC/AICC/BIC; MSE/RMSE/MAE calculation") {
    //    Similar to R's implementation with arima() by MLE.
    //    The difference is that we use OLS insteand of MLE.
    //    The AIC is not suitable for OLS since we did not directly estimate likelihood.
    //    MLE implementation to do. Also, the regularization for OLS?

    //    datain<-read.csv("economics-timeseries.csv")
    //    unemploy<-datain$unemploy
    //    fit<-arima(unemploy,order=c(1,0,0))
    //    summary(fit)
    //    AIC(fit)
    //    BIC(fit)

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val maxLag = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val n = df.count().toInt
    val meanOut = true
    val meanValue = getDouble(df.select(mean(inputCol)).collect()(0).get(0))

    val lr_ar = AutoRegression(inputCol, timeCol, maxLag,
      regParam, standardization, elasticNetParam, withIntercept, meanOut)

    // Fit the model
    val model1 = lr_ar.fit(df)

    val pred1 = model1.transform(df)

    pred1.show(10)

    val intercept = lr_ar.getIntercept()
    val weights = lr_ar.getWeights

    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // get the residuals as (-truth + predicted)
    val residuals = pred1.withColumn("residual", -col("prediction") + col("label")).select("residual")
    residuals.show(10)

    val aic = TimeSeriesUtil.AIC(residuals, maxLag, n)
    val aicc = TimeSeriesUtil.AICc(residuals, maxLag, n)
    val bic = TimeSeriesUtil.BIC(residuals, maxLag, n)
    println(s"AIC value for Lag ${maxLag} is ${aic};\n" +
      s"AICc value for Lag ${maxLag} is ${aicc};\n" +
      s"BIC value for Lag ${maxLag} is ${bic}\n")

    val lmTrainingError = RegressionMetrics("label", "prediction").getMetrics(pred1)
    val rmseSpark = lmTrainingError("rmse")
    val mseSpark = lmTrainingError("mse")
    val maeSpark = lmTrainingError("mae")

    println(s"rmse value for Lag ${maxLag} is ${rmseSpark};\n" +
      s"mse value for Lag ${maxLag} is ${mseSpark};\n" +
      s"mae value for Lag ${maxLag} is ${maeSpark}")
  }

}
