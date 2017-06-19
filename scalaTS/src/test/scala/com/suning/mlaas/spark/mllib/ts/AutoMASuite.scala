package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class AutoMASuite extends SparkSuite {
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

  test("Auto MovingAverage AIC") {
    //    The difference is that we use OLS insteand of MLE.
    //    The AIC is not suitable for OLS since we did not directly estimate likelihood.
    //    MLE implementation to do. Also, the regularization for OLS?

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val q_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aic"
    val earlyStop = false
    val meanOut = false

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Automa = AutoMA(inputCol, timeCol, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)

    val model = lr_Automa.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Automa.getIntercept()
    val weights = lr_Automa.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto MovingAverage BIC") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val q_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "bic"
    val earlyStop = false
    val meanOut = false

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Automa = AutoMA(inputCol, timeCol, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)

    val model = lr_Automa.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Automa.getIntercept()
    val weights = lr_Automa.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto MovingAverage AICC") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val q_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aicc"
    val earlyStop = false
    val meanOut = false

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Automa = AutoMA(inputCol, timeCol, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)

    val model = lr_Automa.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Automa.getIntercept()
    val weights = lr_Automa.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }
  test("Auto MovingAverage AIC with early stop") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val q_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aic"
    val earlyStop = true
    val meanOut = false

    //      Find best model by criterion with earlystop,
    val lr_Automa = AutoMA(inputCol, timeCol, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)

    val model = lr_Automa.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Automa.getIntercept()
    val weights = lr_Automa.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto MovingAverage BIC with early stop") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val q_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "bic"
    val earlyStop = true
    val meanOut = false

    //      Find best model by criterion with earlystop,
    val lr_Automa = AutoMA(inputCol, timeCol, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)


    val model = lr_Automa.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Automa.getIntercept()
    val weights = lr_Automa.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto MovingAverage AICc with early stop") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val q_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aicc"
    val earlyStop = true
    val meanOut = false

    //      Find best model by criterion with earlystop,
    val lr_Automa = AutoMA(inputCol, timeCol, q_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)


    val model = lr_Automa.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Automa.getIntercept()
    val weights = lr_Automa.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }
}
