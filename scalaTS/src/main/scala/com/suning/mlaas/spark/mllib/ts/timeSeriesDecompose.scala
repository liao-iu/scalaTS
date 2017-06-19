package com.suning.mlaas.spark.mllib.ts

//https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/seasonal.py
import breeze.linalg.{DenseVector => BDV}
import com.suning.mlaas.spark.mllib.SQLData.ToRDD
import com.suning.mlaas.spark.mllib.transform.Transformer
import com.suning.mlaas.spark.mllib.util.{Identifiable, SaveLoad}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.rdd.RDDFunctions._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import scala.collection.mutable.{Map => MMap}

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

class TimeSeriesDecompose(override val uid: String, keepOriginal: Boolean,
                          inputCol: String,
                          dateCol:String, period: Int, model: String, filt: Array[Double], two_sided: Boolean)
  extends Transformer(uid, keepOriginal) {

  def this(keepOriginal: Boolean, inputCol: String, dateCol:String, period: Int,
           model: String, filt: Array[Double], two_sided: Boolean) =
    this(Identifiable.randomUID("timeSeriesDecompose"), true, inputCol, dateCol,period, model, filt, two_sided)

  def this(inputCol: String, dateCol:String, period: Int = 12,
           model: String = "additive", filt: Array[Double] = null, two_sided: Boolean = true) =
    this(true, inputCol, dateCol, period, model, filt, two_sided)

  override def transformImpl(df: DataFrame): DataFrame = {
    val obj = new TimeSeriesDecomposeModel(inputCol,dateCol, period, model, filt, two_sided)
    val result = obj.decompose(df)
    result
  }

  override def removeOriginal(df: DataFrame): DataFrame = {
    val obj = new TimeSeriesDecomposeModel(inputCol,dateCol, period, model, filt, two_sided)
    val result = obj.decompose(df).drop(inputCol + "_original")
    result
  }

  override def fitImpl(df: DataFrame): this.type = {
    // nothing to do
    this
  }

  override def save(path: String): Unit = {
    // nothing to do
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    // nothing to do
  }
}


class TimeSeriesDecomposeModel[T](colName: String,
                                  dateCol:String,
                                  period: Int = 12,
                                  model: String = "additive",
                                  filt: Array[Double] = null,
                                  two_sided: Boolean = true)
  extends SaveLoad[TimeSeriesDecompose] with Serializable {

  var kernel: BDV[Double] = _
  var detrended: RDD[Double] = _
  var periodAverage: RDD[Double] = _
  var resid: RDD[Double] = _
  var trendUpdate: RDD[Double] = _

  def getKernel(period: Int): (BDV[Double]) = {
    if (period % 2 == 0) {
      val kernel = BDV.fill(period + 1)(1.0 / period)
      kernel(0) = 0.5 / period
      kernel(period) = 0.5 / period
      kernel
    } else {
      BDV.fill(period)(1.0 / period)
    }
  }

  def toBreezeVector(input: Array[Double]): BDV[Double] = {
    val sizeInput = input.length
    val output = BDV.ones[Double](sizeInput)
    val i = 0
    for (i <- 0 until sizeInput) {
      output(i) = input(i)
    }
    output
  }

  def convolveSmall(inData: BDV[Double], kernel: BDV[Double]): Double = {
    val kernelSize = kernel.size
    val dataSize = inData.size
    var sumValue = 0.0
    for (i <- 0 until dataSize) {
      sumValue += kernel(i) * inData(dataSize - i - 1)
    }
    sumValue
  }

  def takeRangeRDD(input: RDD[Double], start: Int, end: Int): RDD[Double] = {
    val intputTwo: RDD[Double] = input
      .zipWithIndex.filter(x => x._2 > start && x._2 < end).map(x => x._1)
    intputTwo
  }

  def substractRDDTwo(sc: SparkContext,
                      inputOne: RDD[Double],
                      inputTwo: RDD[Double]): RDD[Double] = {
    val inputOneKey = inputOne.zipWithIndex.map { case (value, index) => (index, value) }
    val inputTwoKey = inputTwo.zipWithIndex.map { case (value, index) => (index, value) }
    val result: RDD[Double] = sc.union(inputOneKey, inputTwoKey)
      .reduceByKey((x, y) => x - y).sortByKey()
      .map { case (key, value) => value }
    // inputOneKey.union(inputTwoKey)
    result
  }

  def divideRDDTwo(sc: SparkContext, inputOne: RDD[Double], inputTwo: RDD[Double]): RDD[Double] = {
    val inputOneKey = inputOne.zipWithIndex.map { case (value, index) => (index, value) }
    val inputTwoKey = inputTwo.zipWithIndex.map { case (value, index) => (index, value) }
    val result: RDD[Double] = sc.union(inputOneKey, inputTwoKey)
      .reduceByKey((x, y) => x / y)
      .sortByKey().map { case (key, value) => value }
    // inputOneKey.union(inputTwoKey)
    result
  }

  def average[T](ts: Iterable[T])(implicit num: Numeric[T]) = {
    num.toDouble(ts.sum) / ts.size
  }


  def combineRDDS(inputOne: RDD[Double],
                  inputTwo: RDD[Double],
                  inputThree: RDD[Double], inputFour: RDD[Double], inputFive:RDD[String]): RDD[(Double, Double, Double, Double, String)] = {

    val inputOneKey = inputOne.zipWithIndex.map { case (value, index) => (index, value) }
    val inputTwoKey = inputTwo.zipWithIndex.map { case (value, index) => (index, value) }
    val inputThreeKey = inputThree.zipWithIndex.map { case (value, index) => (index, value) }
    val inputFourKey = inputFour.zipWithIndex.map { case (value, index) => (index, value) }
    val inputFiveKey = inputFive.zipWithIndex.map { case (value, index) => (index, value) }
    val result = inputOneKey
      .join(inputTwoKey)
      .join(inputThreeKey)
      .join(inputFourKey)
      .join(inputFiveKey)
      .sortByKey()
      .map(x => (x._2._1._1._1._1, x._2._1._1._1._2, x._2._1._1._2, x._2._1._2, x._2._2))

    result
  }

  def decompose(df: DataFrame): DataFrame = {
    val sc = df.sqlContext.sparkContext
    require(period > 0, "you must specify a frequency")
    if (filt == null) {
      kernel = getKernel(period)
    }
    else {
      kernel = toBreezeVector(filt)
    }
    // period = 12, colName = "pce"
    val kernelSize = if (period % 2 == 0) period + 1 else period
    val data = ToRDD.toRDDVector(df.select(colName)).map(item => item(0))
    val timeCol = df.select(dateCol).rdd.map(item => item(0).toString)
    val trend: RDD[Double] = data
      .sliding((kernelSize))
      .map(sliceingData => convolveSmall(toBreezeVector(sliceingData), kernel))
    val missingRDDTop: org.apache.spark.rdd.RDD[Double] = df.sqlContext.
      sparkContext.parallelize(Seq.fill(period / 2)(Double.NaN))
    val missingRDDBottom: org.apache.spark.rdd.RDD[Double] = df.sqlContext.
      sparkContext.parallelize(Seq.fill(period / 2)(Double.NaN))
    trendUpdate = sc.union(missingRDDTop, trend, missingRDDBottom)
    model match {
      case "multiplicative" => {
        detrended = divideRDDTwo(sc, data, trendUpdate)
      }
      case "additive" => {
        detrended = substractRDDTwo(sc, data, trendUpdate)
      }
      case _ => throw new IllegalArgumentException(
        s"Not support datatype $model"
      )
    }
    val nobs = data.count
    val removeNANDetrend = takeRangeRDD(detrended, (kernelSize / 2 - 1).asInstanceOf[Int],
      (nobs - kernelSize / 2).asInstanceOf[Int])
    // get the season mean value
    val periodAverages = removeNANDetrend.zipWithIndex.map { case (value, index) => ((index.asInstanceOf[Int] + 6) % period, value) } // for two-sided
    var nullCount = 0
    val seasoanlAver = periodAverages.groupByKey().sortByKey()
      .map(x => (x._1, average(x._2))) // get the mean of each time
    val meanValue = seasoanlAver.map(_._2).sum() / period //-0.46235942420135329
    // val periodAverage = seasoanlAver.map { case (k, v) => v - meanValue } // for the additive
    model match {
      case "multiplicative" => {
        periodAverage = seasoanlAver.map { case (k, v) => v / meanValue } // for the additive      }
      }
      case "additive" => {
        periodAverage = seasoanlAver.map { case (k, v) => v - meanValue }
      }
      case _ => throw new IllegalArgumentException(
        s"Not support datatype $model"
      )
    }
    val repeatTimes = nobs.asInstanceOf[Int] / period.asInstanceOf[Int] + 1
    val dataPeriod = periodAverage.collect() // in case the value of peirod is too big.
    var i = 0
    var dict = Map[Double, Double]()
    // create the dict
    for (i <- 0 until period) {
      dict = dict + (i.asInstanceOf[Double] -> dataPeriod(i))
    }
    val seasonalData = data.zipWithIndex
      .map { case (value, index) => dict.getOrElse(index.asInstanceOf[Double] % period.asInstanceOf[Double], 0).asInstanceOf[Double] }
    val endSeansonalData = takeRangeRDD(seasonalData, (kernelSize / 2 - 1).asInstanceOf[Int], (nobs - kernelSize / 2).asInstanceOf[Int]) // two side
    //get the residual value
    model match {
      case "multiplicative" => {
        resid = divideRDDTwo(sc, removeNANDetrend, endSeansonalData)
      }
      case "additive" => {
        resid = substractRDDTwo(sc, removeNANDetrend, endSeansonalData)
      }
      case _ => throw new IllegalArgumentException(
        s"Not support datatype $model"
      )
    }
    // add the 0s in the beggining of resid and detrend RDD
    val missingRDDTopZero: org.apache.spark.rdd.RDD[Double] = df.sqlContext.sparkContext.parallelize(Seq.fill(period / 2)(0))
    val missingRDDBottomZero: org.apache.spark.rdd.RDD[Double] = df.sqlContext
      .sparkContext
      .parallelize(Seq.fill(period / 2)(0))

    val residFinal = sc.union(missingRDDTopZero, resid, missingRDDBottomZero)
    val trendFinal = sc.union(missingRDDTopZero, trend, missingRDDBottomZero)
    val resFinal = combineRDDS(data, trendFinal, seasonalData, residFinal, timeCol)
    import df.sqlContext.implicits._
    val result = resFinal.toDF(colName + "_original",
      colName + "_trend", colName + "_seasonal", colName + "_residual", colName + "_date")
    result
  }

}
