package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class AutoARSuite extends SparkSuite {
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

  test("Auto AutoRegression AIC") {
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
    val p_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aic"
    val earlyStop = false
    val meanOut = true

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Autoar = AutoAR(inputCol, timeCol, p_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)

    val model = lr_Autoar.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoar.getIntercept()
    val weights = lr_Autoar.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto AutoRegression BIC") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "bic"
    val earlyStop = false
    val meanOut = true

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Autoar = AutoAR(inputCol, timeCol, p_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)

    val model = lr_Autoar.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoar.getIntercept()
    val weights = lr_Autoar.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto AutoRegression AICC") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aicc"
    val earlyStop = false
    val meanOut = true

    //      Find best model by criterion.
    //      If earlystop,
    val lr_Autoar = AutoAR(inputCol, timeCol, p_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)

    val model = lr_Autoar.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoar.getIntercept()
    val weights = lr_Autoar.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }
  test("Auto AutoRegression AIC with early stop") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aic"
    val earlyStop = true
    val meanOut = true

    //      Find best model by criterion with earlystop,
    val lr_Autoar = AutoAR(inputCol, timeCol, p_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)

    val model = lr_Autoar.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoar.getIntercept()
    val weights = lr_Autoar.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto AutoRegression BIC with early stop") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "bic"
    val earlyStop = true
    val meanOut = true

    //      Find best model by criterion with earlystop,
    val lr_Autoar = AutoAR(inputCol, timeCol, p_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)


    val model = lr_Autoar.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoar.getIntercept()
    val weights = lr_Autoar.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }

  test("Auto AutoRegression AICc with early stop") {

    //df.printSchema()
    val regParam = 0
    val withIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val p_Max = 10
    val inputCol = "unemploy"
    val timeCol = "date"

    val crieterion = "aicc"
    val earlyStop = true
    val meanOut = true

    //      Find best model by criterion with earlystop,
    val lr_Autoar = AutoAR(inputCol, timeCol, p_Max,
      regParam, standardization, elasticNetParam, withIntercept, meanOut, crieterion, earlyStop)


    val model = lr_Autoar.fit(df)
    val pred = model.transform(df)


    val intercept = lr_Autoar.getIntercept()
    val weights = lr_Autoar.getWeights
    //
    println(s"Coefficients: ${weights} Intercept: ${intercept}")

    // Fit the model
    pred.show(10)
  }
}
