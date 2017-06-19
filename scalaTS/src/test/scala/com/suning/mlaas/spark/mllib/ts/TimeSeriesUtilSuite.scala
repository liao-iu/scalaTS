package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.SparkSuite
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  **/

class TimeSeriesUtilSuite extends SparkSuite {

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

  test("Auto Correlation function values") {
    df.printSchema()
    df.show()
    val corrs = TimeSeriesUtil.AutoCorrelationFunc(df, "unemploy", "date", numLags = 5)
    corrs.foreach(println)
  }

  test("Auto Correlation function values with decimal") {
    df = df.withColumn("unemploy_d", df("unemploy").cast("decimal"))
    df.printSchema()
    df.show()
    val corrs = TimeSeriesUtil.AutoCorrelationFunc(df, "unemploy_d", "date", numLags = 5)
    corrs.foreach(println)
  }

  test("Yule-Walker Equations lag 5") {
    df.printSchema()
    val test = TimeSeriesUtil.YuleWalker(df, "unemploy", "date", numLags = 5)
    test.foreach(println)
  }

  test("Yule-Walker Equations lag 10") {
    df.printSchema()
    //    R code:
    //    acf(unemploy,lag.max=10,type='partial',plot=F)
    val test = TimeSeriesUtil.YuleWalker(df, "unemploy", "date", numLags = 10)
    test.foreach(println)
    val bounds = TimeSeriesUtil.getBound(df.count)
    println(bounds)
  }

  test("Yule-Walker Equations lag 1") {
    df.printSchema()
    val test = TimeSeriesUtil.YuleWalker(df, "unemploy", "date", numLags = 1)
    test.foreach(println)
  }

  test("Pseudo inverse by SVD") {
    df.printSchema()
    //    Create matrix for inverse
    val newDF = df.select("unemploy", "date")
    val numLag = 5
    val corrs = TimeSeriesUtil.AutoCorrelationFunc(df, "unemploy", "date", numLag, twoDecimal = false)

    val corrslist = corrs.reverse ++ corrs.tail
    val corrslist0 = Vectors.dense(corrslist.slice(from = numLag + 1, until = 2 * numLag + 1))

    val denseData = (0 to numLag toSeq).map(i => {
      Vectors.dense(corrslist.slice(from = numLag - i, until = 2 * numLag - i + 1))
    })

    var denseMat: RowMatrix = new RowMatrix(df.sparkSession.sparkContext.parallelize(denseData, 2))

    val denseDataInverse = TimeSeriesUtil.computeInverse(denseMat)
    println(denseDataInverse)
  }

  //PACF with OLS to fit ar models.
  test("Partial Auto Correlation function values") {
    df.printSchema()
    val pcorrs = TimeSeriesUtil.PartialAutoCorrelationFunc(df, "unemploy", "date", numLags = 5)
    pcorrs.foreach(println)
    val bounds = TimeSeriesUtil.getBound(df.count)
    println(bounds)
  }

  //PACF with Yule-Walker method.
  test("Partial Auto Correlation function values by Yule-Walker") {
    df.printSchema()
    val pcorrs = TimeSeriesUtil.PartialAutoCorrelationFunc(df, "unemploy", "date", numLags = 5, method = "Yule-Walker")
    pcorrs.foreach(println)
    val bounds = TimeSeriesUtil.getBound(df.count)
    println(bounds)
  }

  test("time series time combination") {
    df.printSchema()
    var Lags = TimeSeriesUtil.LagCombination(df, inputCol = "unemploy", timeCol = "date", p = 5)
    Lags.show(100)
  }

  test("time series time combination by taking mean out") {
    df.printSchema()
    val Lags = TimeSeriesUtil.LagCombination(df, inputCol = "unemploy", timeCol = "date", p = 5,
      lagsOnly = true, meanOut = true)
    Lags.show(100)
  }

  test("time series time differencing combination") {
    df.printSchema()
    var Diffs = TimeSeriesUtil.DiffCombination(df, inputCol = "unemploy", timeCol = "date", maxLag = 5, diff = 1, lagsOnly = false)
    Diffs.show(100)
  }
  //
  //  test("time series time differencing 2 combination") {
  //    df.printSchema()
  //    var Diffs = TimeSeriesUtil.DiffCombination(df, inputCol = "unemploy", timeCol = "date", maxLag = 5, diff = 2)
  //    Diffs.show(100)
  //  }

}
