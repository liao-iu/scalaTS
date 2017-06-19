package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class AutoARIMASuite extends SparkSuite {
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

  test("Auto ARIMA AIC") {
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p_Max = 5
    val d_Max = 1
    val q_Max = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aic"
    val earlyStop = false
    val meanOut = true

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Autoarima = AutoARIMA(inputCol, timeCol, p_Max, d_Max, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion)

    val model = lr_Autoarima.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoarima.getIntercept()
    val weights = lr_Autoarima.getWeights
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto ARIMA BIC") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p_Max = 5
    val d_Max = 1
    val q_Max = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "bic"
    val earlyStop = false
    val meanOut = true

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Autoarima = AutoARIMA(inputCol, timeCol, p_Max, d_Max, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion)

    val model = lr_Autoarima.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoarima.getIntercept()
    val weights = lr_Autoarima.getWeights
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto ARIMA AICC") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p_Max = 5
    val d_Max = 1
    val q_Max = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aicc"
    val earlyStop = false
    val meanOut = true

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Autoarima = AutoARIMA(inputCol, timeCol, p_Max, d_Max, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion)

    val model = lr_Autoarima.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoarima.getIntercept()
    val weights = lr_Autoarima.getWeights
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

}
