package com.suning.mlaas.spark.mllib.regression

import com.suning.mlaas.spark.mllib.SparkSuite
import com.suning.mlaas.spark.mllib.metric.RegressionMetrics
import com.suning.mlaas.spark.mllib.util.TestingUtil._
import org.apache.spark.ml.linalg.Vectors

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

class LinearRegressionSuite extends SparkSuite {

  override def beforeAll(): Unit = {
    super.beforeAll()

    df = sparkSession.read
      .format("com.databricks.spark.csv")
      .option("header", "true") // Use first line of all files as header
      .option("inferSchema", "true") // Automatically infer data types
      .load("data/mllib/sample_linear_regression_data.csv")

  }

  test("Simple linear regression with no regularization"){

    val features = Array("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10")
    val label = "y"

    val regParam = 0
    val fitIntercept = true
    val standardization = false
    val elasticNetParam = 0
    val maxIter = 100
    val tol = 1E-6

    val lr = LinearRegression(features, label, regParam, fitIntercept, standardization, elasticNetParam, maxIter, tol)

    // Fit the model
    val linearModel = lr.fit(df)

    // get intercept
    val interceptR = 0.142285583
    val intercept = linearModel.getIntercept()
    //println(intercept)
    assert(interceptR ~= intercept relTol 1E-3)

    // get weights
    val weightsR = Vectors.dense(Array(
      0.007335071, 0.831375758, -0.809530795, 2.441191687,
      0.519171380, 1.153459190, -0.298912411, -0.512851419,
      -0.619712827, 0.695615180))
    val weights = linearModel.getWeights()
    //print(weights.toString)
    assert(weights ~== weightsR relTol 1E-3)

    // Make predictions
    val predResult = linearModel.transform(df)


    val lmtest1 = RegressionMetrics(label, "prediction").getMetrics(predResult)


    // regression evaluation metric: root mean squared error
    // compute RMSE
    val rmseR = 10.16309

    val rmseSpark1 = lmtest1("rmse")
    assert(rmseR ~= rmseSpark1 relTol 1E-3)
  }

  test("Linear regression with intercept with L1 regularization") {

    val features = Array("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10")
    val label = "y"

    val regParam = 0.57
    val fitIntercept = true
    val standardization_1 = true
    val standardization_2 = false
    val elasticNetParam = 1.0
    val maxIter = 100
    val tol = 1E-6

    // Fit the model
    val lr_1 = LinearRegression(features, label, regParam, fitIntercept, standardization_1, elasticNetParam, maxIter, tol)
    val lr_2 = LinearRegression(features, label, regParam, fitIntercept, standardization_2, elasticNetParam, maxIter, tol)
    val model1 = lr_1.fit(df)
    val model2 = lr_2.fit(df)

    // compare coefficients
    val interceptR1 = 0.1874801
    val weightsR1 = Vectors.sparse(10, Array(3, 5), Array(1.2808573, 0.1897744))
    val interceptR2 = 0.2328537
    val weightsR2 = Vectors.sparse(10, Array(3), Array(0.4165744))

    assert(model1.getIntercept() ~== interceptR1 relTol 1E-2)
    assert(model1.getWeights() ~= weightsR1 relTol 1E-2)
    assert(model2.getIntercept() ~== interceptR2 relTol 1E-2)
    assert(model2.getWeights() ~= weightsR2 relTol 1E-2)

    // make predictions
    val pred1 = model1.transform(df)
    val pred2 = model2.transform(df)


    val lml1test1 = RegressionMetrics(label, "prediction").getMetrics(pred1)


    // regression evaluation metric: root mean squared error
    // compute RMSE
    val rmseR_1 = 10.23842

    val rmseSpark1 = lml1test1("rmse")
    assert(rmseR_1 ~= rmseSpark1 relTol 1E-3)



    val lml1test2 = RegressionMetrics(label, "prediction").getMetrics(pred2)


    // regression evaluation metric: root mean squared error
    // compute RMSE
    val rmseR_2 = 10.28199
    val rmseSpark2 = lml1test2("rmse")
    assert(rmseR_1 ~= rmseSpark1 relTol 1E-3)
  }

  test("linear regression with intercept with L2 regularization") {

    val features = Array("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10")
    val label = "y"

    val regParam = 2.3
    val fitIntercept = true
    val standardization_1 = true
    val standardization_2 = false
    val elasticNetParam = 0
    val maxIter = 100
    val tol = 1E-6

    // Fit the model
    val lr_1 = LinearRegression(features, label, regParam, fitIntercept, standardization_1, elasticNetParam, maxIter, tol)
    val lr_2 = LinearRegression(features, label, regParam, fitIntercept, standardization_2, elasticNetParam, maxIter, tol)
    val model1 = lr_1.fit(df)
    val model2 = lr_2.fit(df)

    // compare coefficients
    val interceptR1 = 0.16526297
    val weightsR1 = Vectors.dense(0.02379074, 0.64001298, -0.65035141, 1.97408164, 0.40661341, 0.95028797, -0.25696602, -0.41903179, -0.51404167, 0.55127621)
    val interceptR2 = 0.19462306
    val weightsR2 = Vectors.dense(0.03182069, 0.43162679, -0.46244090, 1.36656677, 0.28403477, 0.68472236, -0.20288277, -0.30353884, -0.37064210, 0.39558533)

    assert(model1.getIntercept() ~== interceptR1 relTol 1E-2)
    assert(model1.getWeights() ~= weightsR1 relTol 1E-2)
    assert(model2.getIntercept() ~== interceptR2 relTol 1E-2)
    assert(model2.getWeights() ~= weightsR2 relTol 1E-2)

    // make predictions
    val pred1 = model1.transform(df)
    val pred2 = model2.transform(df)



    val orftest1 = RegressionMetrics(label, "prediction").getMetrics(pred1)


    // regression evaluation metric: root mean squared error
    // compute RMSE
    val rmseR1 = 10.16843

    val rmseSpark1 = orftest1("rmse")
    assert(rmseR1 ~= rmseSpark1 relTol 1E-3)


    val orftest2 = RegressionMetrics(label, "prediction").getMetrics(pred2)

    val rmseR2 = 10.19031
    val rmseSpark2 = orftest2("rmse")
    assert(rmseR2 ~= rmseSpark2 relTol 1E-3)

  }

  test("linear regression with intercept with ElasticNet regularization") {

    val features = Array("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10")
    val label = "y"

    val regParam = 1.6
    val fitIntercept = true
    val standardization_1 = true
    val standardization_2 = false
    val elasticNetParam = 0.3
    val maxIter = 100
    val tol = 1E-6

    // Fit the model
    val lr_1 = LinearRegression(features, label, regParam, fitIntercept, standardization_1, elasticNetParam, maxIter, tol)
    val lr_2 = LinearRegression(features, label, regParam, fitIntercept, standardization_2, elasticNetParam, maxIter, tol)
    val model1 = lr_1.fit(df)
    val model2 = lr_2.fit(df)

    // compare coefficients
    val interceptR1 = 0.1890835
    val weightsR1 = Vectors.sparse(10, Array(3, 5), Array(1.3049812, 0.3162831))
    val interceptR2 = 0.2265281
    val weightsR2 = Vectors.sparse(10, Array(3), Array(0.5262077))

    assert(model1.getIntercept() ~== interceptR1 relTol 1E-2)
    assert(model1.getWeights() ~= weightsR1 relTol 1E-2)
    assert(model2.getIntercept() ~== interceptR2 relTol 1E-2)
    assert(model2.getWeights() ~= weightsR2 relTol 1E-2)

    // make predictions
    val pred1 = model1.transform(df)
    val pred2 = model2.transform(df)


    val orftest1 = RegressionMetrics(label, "prediction").getMetrics(pred1)



    // regression evaluation metric: root mean squared error
    // compute RMSE
    val rmseR1 = 10.23399

    val rmseSpark1 = orftest1("rmse")
    assert(rmseR1 ~= rmseSpark1 relTol 1E-3)

    val orftest2 = RegressionMetrics(label, "prediction").getMetrics(pred2)

    val rmseR2 = 10.27609
    val rmseSpark2 = orftest2("rmse")
    assert(rmseR2 ~= rmseSpark2 relTol 1E-3)

  }


}
