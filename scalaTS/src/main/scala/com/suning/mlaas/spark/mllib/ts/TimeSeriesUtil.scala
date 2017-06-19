package com.suning.mlaas.spark.mllib.ts

import java.math.{BigDecimal => jBigDec}
import java.text.DecimalFormat
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix,Vector => OldVector, DenseVector => OldDenseVector, Vectors => OldVectors}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import scala.math.BigDecimal

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */

object TimeSeriesUtil {
  //only print two decimals
  val formatter = new DecimalFormat("#.###")

  def getBound(count: Long): (Double, Double) = {
    val bound = 1.96 / math.sqrt(count)
    (-bound, bound)
  }

  def getDouble(value: Any): Double = {
    //    println(value.getClass)
    value match {
      case s: Short => s.toDouble
      case i: Int => i.toDouble
      case l: Long => l.toDouble
      case f: Float => f.toDouble
      case d: Double => d
      case s: String => s.toDouble
      case jbd: jBigDec => jbd.doubleValue
      case bd: BigDecimal => bd.doubleValue
      case _ => 0.0
    }
  }

  val toDouble = udf((value: Any) => getDouble(value))


  def AutoCorrelationFunc(df: DataFrame, inputCol: String, timeCol: String, numLags: Int,
                          twoDecimal: Boolean = true):
  Array[Double] = {

    val newDF = df.select(inputCol, timeCol)

    val meanValue = getDouble(df.select(mean(inputCol)).collect()(0).get(0))
    val varianceValue = df.stat.cov(inputCol, inputCol)
    val count = df.count()

    // udf to calculate the covariance
    val transfermap = udf((col1: Any, col2: Any) => {
      ((getDouble(col1) - meanValue) * (getDouble(col2) - meanValue))
    })

    val corrs = (0 to numLags toArray).map(i => {
      val lagCol = inputCol + "_lag_" + i

      val output = TimeSeriesLag(inputCol, lagCol, timeCol, i)
        .transform(newDF).filter(col(inputCol + "_lag_" + i).isNotNull)

      val diff1 = output.withColumn("diff1", transfermap(col(inputCol), col(lagCol)))
      // correct auto-correlation function
      if (twoDecimal) {
        formatter.format(getDouble(diff1.select(sum("diff1")).collect()(0).get(0))
          / ((count - 1) * varianceValue)).toDouble
      }
      else {
        getDouble(diff1.select(sum("diff1")).collect()(0).get(0)) / ((count - 1) * varianceValue)
      }
    })
    corrs
  }

  //  Yule-Walker eqns. for PACF func.
  def YuleWalker(df: DataFrame, inputCol: String, timeCol: String, numLags: Int):
  Array[Double] = {
    val newDF = df.select(inputCol, timeCol)
    val corrs = AutoCorrelationFunc(df, inputCol, timeCol, numLags, twoDecimal = false)

    val corrslist = corrs.reverse ++ corrs.tail
    var corrslistMap = corrslist
    //    corrslistMap.foreach(println)

    val YWpcorrs = (1 to (numLags - 1) toArray).map(i => {

      corrslistMap = corrslist.slice(from = numLags - i, until = numLags + i + 1)
      //   corrslistMap.foreach(println)
      val corrslist0 = OldVectors.dense(corrs.slice(from = 1, until = i + 2))

      val denseData = (0 to i toSeq).map(j => {
        org.apache.spark.mllib.linalg.Vectors.dense(corrslistMap.slice(from = i - j, until = 2 * i - j + 1))
      })

      val denseMat: RowMatrix = new RowMatrix(df.sqlContext.sparkContext.parallelize(denseData, 2))
      //       denseMat.rows.collect.foreach(println)

      val InversedenseMat = computeInverse(denseMat)

      InversedenseMat.multiply(corrslist0).values(i)
    })
    Array(corrs(1)) ++ YWpcorrs
  }

  // Calculate the inversion of a matrix using SVD. Note that V is not distributed.
  // Future work is to make it distributed.
  def computeInverse(X: RowMatrix): OldDenseMatrix = {
    val nCoef = X.numCols.toInt
    val svd = X.computeSVD(nCoef, computeU = true)
    if (svd.s.size < nCoef) {
      sys.error(s"Singular matrix")
    }

    // Create the inv diagonal matrix from S
    val invS = OldDenseMatrix.diag(new OldDenseVector(svd.s.toArray.map(x => math.pow(x, -1))))
    // U cannot be a RowMatrix
    val U = new OldDenseMatrix(svd.U.numRows().toInt, svd.U.numCols().toInt,
      svd.U.rows.collect.flatMap(x => x.toArray))

    // If you could make V distributed, then this may be better. However its alreadly local...so maybe this is fine.
    val V = svd.V
    // inv(X) = V*inv(S)*transpose(U)  --- the U is already transposed.
    (V.multiply(invS)).multiply(U)
  }


  def PartialAutoCorrelationFunc(df: DataFrame, inputCol: String, timeCol: String, numLags: Int):
  Array[Double] = {
    val newDF = df.select(inputCol, timeCol)

    // regression parameters
    val regParam = 0
    val withIntercept = false
    val standardization = false
    val elasticNetParam = 0
    val meanOut = false

    val pcorrs = (0 to numLags - 1 toArray).map(i => {

      val lr_ar = AutoRegression(inputCol, timeCol, i + 1, regParam,
        standardization, elasticNetParam, withIntercept)

      lr_ar.fit(newDF)
      val weights = lr_ar.getWeights()

      formatter.format(weights(i)).toDouble
    })
    pcorrs
  }

  def PartialAutoCorrelationFunc(df: DataFrame, inputCol: String, timeCol: String, numLags: Int,
                                 method: String = ""): Array[Double] = {
    if (method == "Yule-Walker") {
      YuleWalker(df, inputCol, timeCol, numLags)
    } else {
      PartialAutoCorrelationFunc(df, inputCol, timeCol, numLags)
    }
  }

  def LagCombination(df: DataFrame, inputCol: String, timeCol: String, p: Int): DataFrame = {
    val lag0Col = inputCol + "_lag_0"

    val newDF = df.withColumn(lag0Col, toDouble(df(inputCol)))

    newDF.persist()

    val outputDF = (1 to p).foldLeft(newDF)((df, i) => {
      val lagCol = inputCol + "_lag_" + i
      TimeSeriesLag(lag0Col, lagCol, timeCol, i).transform(df)
    })
    newDF.unpersist()
    outputDF
  }

  def LagCombination(df: DataFrame, inputCol: String, timeCol: String, p: Int,
                     lagsOnly: Boolean): DataFrame = {
    if (lagsOnly) {
      LagCombination(df.select(inputCol, timeCol), inputCol, timeCol, p)
        .drop(inputCol)
    }
    else {
      LagCombination(df, inputCol, timeCol, p)
    }
  }

  def LagCombination(df: DataFrame, inputCol: String, timeCol: String, p: Int,
                     lagsOnly: Boolean = true, meanOut: Boolean = false): DataFrame = {
    if (meanOut) {
      val meanOutCol = inputCol + "_meanOut"
      val meanValue = getDouble(df.select(mean(inputCol)).collect()(0).get(0))
      val meanOut = udf((col: Double) => {
        (col - meanValue)
      })
      val meanOutDF = df.withColumn(meanOutCol, meanOut(col(inputCol)))
      LagCombination(meanOutDF, meanOutCol, timeCol, p, lagsOnly)
    }
    else {
      LagCombination(df, inputCol, timeCol, p, lagsOnly)
    }
  }

  def DiffCombination(df: DataFrame, inputCol: String, timeCol: String,
                      maxLag: Int, diff: Int = 1, lagsOnly: Boolean = true):
  DataFrame = {
    require(diff == 1 || diff == 2, "diff must be 1 or 2")
    val lag0Col = inputCol + "_lag_0"
    val lag1Col = inputCol + "_lag_1"
    val diff1Col = inputCol + "_diff_1"
    val diff1DF = LagCombination(df, inputCol, timeCol, 1, lagsOnly)
      .filter(col(inputCol + "_lag_" + 1).isNotNull)
      .withColumn(diff1Col, col(lag0Col) - col(lag1Col)).drop(lag0Col).drop(lag1Col)

    if (diff == 1)
      LagCombination(diff1DF, diff1Col, timeCol, maxLag, lagsOnly)
        .filter(col(diff1Col + "_lag_" + maxLag).isNotNull)
    else {
      val diff1lag0Col = diff1Col + "_lag_0"
      val diff1lag1Col = diff1Col + "_lag_1"
      val diff2Col = inputCol + "_diff_2"
      val diff2DF = LagCombination(diff1DF, diff1Col, timeCol, 1, lagsOnly)
        .filter(col(diff1Col + "_lag_" + 1).isNotNull)
        .withColumn(diff2Col, col(diff1lag0Col) - col(diff1lag1Col)).drop(diff1lag0Col).drop(diff1lag1Col)

      LagCombination(diff2DF, diff2Col, timeCol, maxLag, lagsOnly)
        .filter(col(diff2Col + "_lag_" + maxLag).isNotNull)
    }
  }

  //combine lags through TimeSeriesLag function for ma modelling.
  def LagCombinationMA(df: DataFrame, inputCol: String,
                       residualCol: String, timeCol: String, labelCol: String, q: Int):
  DataFrame = {
    val newDF = df.select(residualCol, timeCol, labelCol, inputCol + "_lag_" + 0)
    LagCombination(newDF, residualCol, timeCol, q, lagsOnly = false, meanOut = false)
  }

  def LagCombinationARMA(df: DataFrame, inputCol: String, residualCol: String,
                         timeCol: String, predCol: String, p: Int, q: Int, d: Int = 0,
                         arimaFeature: Boolean = false):

  DataFrame = {
    val r = 0 to p toList
    var prefix = ""
    var newDF = df
    // remember all ar features names and transfer to String.
    if (arimaFeature) {
      prefix = "_diff_" + d
      var features_ar = r.map(inputCol + prefix + "_lag_" + _) :+ residualCol :+ timeCol :+ predCol
      //pick all values including ar's lags.
      newDF = df.select(inputCol, features_ar: _*)
    } else {
      //    var prefix = "_meanOut"
      var features_ar = r.map(inputCol + prefix + "_lag_" + _) :+ residualCol :+ timeCol :+ predCol
      //pick all values including ar's lags.
      newDF = df.select(inputCol, features_ar: _*)
    }
    //      df.select(inputCol, features_ar: _*)

    //    newDF.show(10)

    //    transfer the original column lag0 to double


    val lag0Col = residualCol + "_lag_0"

    val newDF_Double = newDF.withColumn(lag0Col, toDouble(newDF(residualCol)))
    //newDF_Double.printSchema()

    var lagcol: Array[String] = Array[String]()

    lagcol = new Array[String](q + 1)

    lagcol(0) = lag0Col

    lagcol(1) = residualCol + "_lag_" + (1)

    val lag = TimeSeriesLag(lagcol(0), lagcol(1), timeCol, 1)

    var output = lag.transform(newDF_Double)

    var i = 2

    while (i <= q) {

      output.persist()

      lagcol(i) = residualCol + "_lag_" + (i)

      val lags = TimeSeriesLag(lagcol(0), lagcol(i), timeCol, i)

      output = lags.transform(output)

      i += 1
    }

    output
  }

  /*
  Calculate the AIC\AICC\BIC of an AR model. Take "residual" column and # of parameters. Prepare for auto.ar(auto.arima).
   */
  def AIC(df: DataFrame, p: Int, n: Int):
  Double = {
    //    val n = df.count()
    val res = df.rdd.map(r => r.getAs[Double](0) * r.getAs[Double](0)).sum()
    val k = (p + 1).toDouble
    //  (p + 1) is that a model has 1 independent variable.
    val aicValue = n * (math.log(2 * math.Pi) + math.log(res / n) + 1) + 2 * k
    aicValue
    //  res
  }

  def AICc(df: DataFrame, p: Int, n: Int):
  Double = {
    //    val n = df.count()
    val aicValue = TimeSeriesUtil.AIC(df, p, n)
    val k = (p + 1).toDouble
    val aiccValue = aicValue + 2 * (k) * (k + 1) / (n - k - 1)
    aiccValue
  }

  def BIC(df: DataFrame, p: Int, n: Int):
  Double = {
    //    val n = df.count()
    val res = df.rdd.map(r => r.getAs[Double](0) * r.getAs[Double](0)).sum()
    val k = (p + 1).toDouble
    //  (p + 1) is that a model has 1 independent variable.
    val bicValue = n * (math.log(2 * math.Pi) + math.log(res / n) + 1) + k * math.log(n)
    bicValue
  }

  def tsFitDotProduct(listDF: List[Double], numHead: Int, p: Int,
                      intercept: Double, weights: Vector, meanValue: Double = 0.0): List[Double] = {

    var listdf = listDF

    (0 until numHead).foreach {
      j => {
        val vec = Vectors.dense(listdf.slice(0, p).toArray)
        var result = 0.0
        (0 until p).foreach(
          i => {
            result += vec(i) * weights(i)
          }
        )
        result = result + intercept

        listdf = result :: listdf
      }
    }

    listdf.map(i => i + meanValue)
  }

  //  Time Series forecasting for ar process
  def tsForecastAR(df: DataFrame, numHead: Int, inputCol: String, timeCol: String, p: Int,
                   intercept: Double, weights: Vector): List[Double] = {
    val prefix = ""
    val lag = "_lag_"
    var listDF = df.orderBy(desc(timeCol)).select(inputCol + prefix + lag + 0)
      .limit(p).collect().map(_.getDouble(0)).toList

    var listPrediction = listDF
    listPrediction = tsFitDotProduct(listDF, numHead, p, intercept, weights, meanValue = 0.0)

    val prediction = listPrediction.slice(0, numHead).reverse
    prediction
  }

  def tsForecastAR(df: DataFrame, numHead: Int, inputCol: String, timeCol: String, p: Int,
                   intercept: Double, weights: Vector, meanOut: Boolean, meanValue: Double): List[Double] = {
    val prefix = if (meanOut) "_meanOut" else ""
    val lag = "_lag_"
    var listDF = df.orderBy(desc(timeCol)).select(inputCol + prefix + lag + 0)
      .limit(p).collect().map(_.getDouble(0)).toList

    var listPrediction = listDF
    if (meanOut) {
      listPrediction = tsFitDotProduct(listDF, numHead, p, intercept, weights, meanValue = meanValue)
    } else {
      listPrediction = tsFitDotProduct(listDF, numHead, p, intercept, weights, meanValue = 0.0)
    }

    val prediction = listPrediction.slice(0, numHead).reverse
    prediction
  }


  //  Time Series forecasting for YuleWalker process
  def tsForecastYuleWalker(df: DataFrame, numHead: Int, inputCol: String, timeCol: String, p: Int,
                           weights: Vector, meanValue: Double): List[Double] = {
    val prefix = ""
    val lag = "_lag_"
    var listDF = df.orderBy(desc(timeCol)).select(inputCol + prefix + lag + 0)
      .limit(p).collect().map(_.getDouble(0)).toList

    listDF = listDF.map(i => i - meanValue)

    var listPrediction = tsFitDotProduct(listDF, numHead, p, intercept = 0.0, weights, meanValue = meanValue)


    val prediction = listPrediction.slice(0, numHead).reverse
    prediction
  }

  def tsForecastMA(df: DataFrame, numHead: Int, inputCol: String, timeCol: String, q: Int,
                   intercept: Double, weights: Vector): List[Double] = {
    val prefix = "residual"
    val lag = "_lag_"
    var listDF = df.orderBy(desc(timeCol)).select(prefix + lag + 0)
      .limit(q).collect().map(_.getDouble(0)).toList

    var listPrediction = listDF
    listPrediction = tsFitDotProduct(listDF, q + 1, q, 0.0, weights, meanValue = 0.0)

    var prediction = listPrediction.slice(0, q + 1).reverse
    prediction = prediction.map(i => i + intercept)

    ((q + 2) to numHead) foreach { _ =>
      prediction = prediction :+ prediction(q)
    }
    prediction
  }

  def tsForecastARMA(df: DataFrame, numHead: Int, inputCol: String, timeCol: String, p: Int, q: Int,
                     intercept: Double, weights: Vector): List[Double] = {
    if (p == 0 || q == 0) {
      if (q == 0) {
        tsForecastAR(df, numHead, inputCol, timeCol, p,
          intercept, weights)
      }
      else {
        tsForecastMA(df, numHead, inputCol, timeCol, q,
          intercept, weights)
      }
    }
    else {
      tsForecastARMAModel(df, numHead, inputCol, timeCol, p, q,
        intercept, weights)
    }
  }

  def tsForecastARMAModel(df: DataFrame, numHead: Int, inputCol: String, timeCol: String, p: Int,
                          q: Int, intercept: Double, weights: Vector): List[Double] = {

    var prefix = ""
    val lag = "_lag_"
    var listDFar = df.orderBy(desc(timeCol)).select(inputCol + prefix + lag + 0)
      .limit(p).collect().map(_.getDouble(0)).toList

    var listPredictionAR = listDFar

    val weightsAR = Vectors.dense(weights.toArray.slice(0, p))
    val weightsMA = Vectors.dense(weights.toArray.slice(p, p + q))

    listPredictionAR = tsFitDotProduct(listDFar, numHead, p, intercept, weightsAR, meanValue = 0.0)

    val predictionAR = listPredictionAR.slice(0, numHead).reverse

    prefix = "residual"
    var listDFma = df.orderBy(desc(timeCol)).select(prefix + lag + 0)
      .limit(q).collect().map(_.getDouble(0)).toList

    var listPredictionMA = listDFma
    listPredictionMA = tsFitDotProduct(listDFma, q + 1, q, 0.0, weightsMA, meanValue = 0.0)

    var predictionMA = listPredictionMA.slice(0, q + 1).reverse
    //    predictionMA = predictionMA.map(i => i + intercept)

    ((q + 2) to numHead) foreach { _ =>
      predictionMA = predictionMA :+ 0.0
    }
    (predictionAR, predictionMA).zipped.map(_ + _)
  }

  //  def LagCombinationARIMA(df: DataFrame, inputCol: String, residualCol: String,timeCol: String, predCol: String, p: Int, d: Int, q: Int):
  //
  //  DataFrame = {
  //
  //    require(d == 1 || d == 2, "diff must be 1 or 2")
  //
  //    val lag0Col = inputCol + "_lag_0"
  //    val lag1Col = inputCol + "_lag_1"
  //    val diff1Col = inputCol + "_diff_1"
  //    val diff1DF = LagCombination(df, inputCol, timeCol, 1)
  //      .filter(col(inputCol + "_lag_" + 1).isNotNull)
  //      .withColumn(diff1Col, col(lag0Col) - col(lag1Col)).drop(lag0Col).drop(lag1Col)
  //
  //    if (d == 1)
  //      LagCombination(diff1DF, diff1Col, timeCol, maxLag).filter(col(diff1Col + "_lag_" + maxLag).isNotNull)
  //    else {
  //      val diff1lag0Col = diff1Col + "_lag_0"
  //      val diff1lag1Col = diff1Col + "_lag_1"
  //      val diff2Col = inputCol + "_diff_2"
  //      val diff2DF = LagCombination(diff1DF, diff1Col, timeCol, 1)
  //        .filter(col(diff1Col + "_lag_" + 1).isNotNull)
  //        .withColumn(diff2Col, col(diff1lag0Col) - col(diff1lag1Col)).drop(diff1lag0Col).drop(diff1lag1Col)
  //
  //      LagCombination(diff2DF, diff2Col, timeCol, maxLag).filter(col(diff2Col + "_lag_" + maxLag).isNotNull)
  //    }
  //    // remember all ar features names and transfer to String.
  //
  //
  //
  //
  //
  //    val r = 0 to p toList
  //    //    var prefix = "_meanOut"
  //    var prefix = "_diff_" + d
  //
  //    val features_ar = r.map(inputCol + prefix + "_lag_" + _)  :+ residualCol :+ timeCol :+ predCol
  //
  //
  //    //pick all values including ar's lags.
  //    val newDF = df.select(inputCol + prefix + "_lag_" + 0, features_ar: _*)
  //
  //    //    newDF.show(10)
  //
  //    //    transfer the original column lag0 to double
  //
  //    val toDouble = udf[Double, String](_.toDouble)
  //
  //    val lag0Col = residualCol + prefix + "_lag_0"
  //
  //    val newDF_Double = newDF.withColumn(lag0Col, toDouble(newDF(residualCol)))
  //    //newDF_Double.printSchema()
  //
  //    var lagcol: Array[String] = Array[String]()
  //
  //    lagcol = new Array[String](q + 1)
  //
  //    lagcol(0) = lag0Col
  //
  //    lagcol(1) = residualCol + prefix + "_lag_" + (1)
  //
  //    val lag = TimeSeriesLag(lagcol(0), lagcol(1), timeCol, 1)
  //
  //    var output = lag.transform(newDF_Double)
  //
  //    var i = 2
  //
  //    while (i <= q) {
  //
  //      output.persist()
  //
  //      lagcol(i) = residualCol + prefix + "_lag_" + (i)
  //
  //      val lags = TimeSeriesLag(lagcol(0), lagcol(i), timeCol, i)
  //
  //      output = lags.transform(output)
  //
  //      i += 1
  //    }
  //
  //    output
  //  }

}
