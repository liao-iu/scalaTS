package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class AutoARMASuite extends SparkSuite {
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

  test("Auto ARMA AIC") {
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p_Max = 5
    val q_Max = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aic"
    val earlyStop = false
    val meanOut = true

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Autoarma = AutoARMA(inputCol, timeCol, p_Max, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, crieterion)

    val model = lr_Autoarma.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoarma.getIntercept()
    val weights = lr_Autoarma.getWeights
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto ARMA BIC") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p_Max = 5
    val q_Max = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "bic"
    val earlyStop = false
    val meanOut = true

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Autoarma = AutoARMA(inputCol, timeCol, p_Max, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, crieterion)

    val model = lr_Autoarma.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoarma.getIntercept()
    val weights = lr_Autoarma.getWeights
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto ARMA AICC") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p_Max = 5
    val q_Max = 5
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aicc"
    val earlyStop = false
    val meanOut = true

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Autoarma = AutoARMA(inputCol, timeCol, p_Max, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, crieterion)

    val model = lr_Autoarma.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoarma.getIntercept()
    val weights = lr_Autoarma.getWeights
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

}
