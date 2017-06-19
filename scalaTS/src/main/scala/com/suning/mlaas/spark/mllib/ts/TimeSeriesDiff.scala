package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.transform.Transformer
import com.suning.mlaas.spark.mllib.ts.TimeSeriesUtil.toDouble
import com.suning.mlaas.spark.mllib.util.{Identifiable, SaveLoad}
import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
class TimeSeriesDiff(override val uid: String, keepOriginal: Boolean,
                     inputCol: String, outputCol: String, timeCol: String, diff: Int, lag: Int)
  extends Transformer(uid, keepOriginal) {

  def this(keepOriginal: Boolean, inputCol: String, outputCol: String, timeCol: String, diff: Int,
           lag: Int) =
    this(Identifiable.randomUID("TimeSeriesDiff"), keepOriginal, inputCol, outputCol, timeCol,
      diff, lag)

  def this(inputCol: String, outputCol: String, timeCol: String, diff: Int, lag: Int) =
    this(true, inputCol, outputCol, timeCol, diff, lag)

  def this(inputCol: String, timeCol: String, diff: Int, lag: Int) =
    this(inputCol, inputCol + "_lag_" + lag + "_diff_" + diff, timeCol, diff, lag)


  override def transformImpl(df: DataFrame): DataFrame = {
    require(diff == 1 || diff == 2, "diff must be 1 or 2")

    val lagCol = inputCol + "_lag_" + lag
    val diff1Col = lagCol + "_diff_1"
    val diff1DF = TimeSeriesLag(inputCol, timeCol, lag).transform(df)
      .filter(col(inputCol + "_lag_" + lag).isNotNull)
      .withColumn(diff1Col, toDouble(col(inputCol) - col(lagCol))).drop(lagCol)

    //first order
    if (diff == 1)
      diff1DF.withColumnRenamed(diff1Col, outputCol)
    //second order
    else {
      val diff1lagCol = diff1Col + "_lag_" + 1
      val diff2Col = inputCol + "_diff_2"
      val diff2DF = TimeSeriesLag(diff1Col, timeCol, 1).transform(diff1DF)
        .filter(col(diff1Col + "_lag_" + 1).isNotNull)
        .withColumn(diff2Col, col(diff1Col) - col(diff1lagCol)).drop(diff1lagCol).drop(diff1Col)
      diff2DF.withColumnRenamed(diff2Col, outputCol)
    }
  }

  override def removeOriginal(df: DataFrame): DataFrame = {
    df.drop(inputCol)
  }

  override def fitImpl(df: DataFrame): this.type = {
    // nothing to do
    this
  }

  override def save(path: String): Unit = {
    TimeSeriesDiff.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    TimeSeriesDiff.saveHDFS(sc, this, path)
  }
}

object TimeSeriesDiff extends SaveLoad[TimeSeriesDiff] {
  def apply(uid: String, keepOriginal: Boolean, inputCol: String, outputCol: String,
            timeCol: String, diff: Int, lag: Int):
  TimeSeriesDiff = new TimeSeriesDiff(uid, keepOriginal, inputCol, outputCol, timeCol, diff, lag)

  def apply(keepOriginal: Boolean, inputCol: String, outputCol: String, timeCol: String,
            diff: Int, lag: Int):
  TimeSeriesDiff = new TimeSeriesDiff(keepOriginal, inputCol, outputCol, timeCol, diff, lag)

  def apply(inputCol: String, outputCol: String, timeCol: String, diff: Int, lag: Int):
  TimeSeriesDiff = new TimeSeriesDiff(inputCol, outputCol, timeCol, diff, lag)

  def apply(inputCol: String, timeCol: String, diff: Int, lag: Int):
  TimeSeriesDiff = new TimeSeriesDiff(inputCol, timeCol, diff, lag)

  def apply(inputCol: String, timeCol: String, diff: Int):
  TimeSeriesDiff = new TimeSeriesDiff(inputCol, timeCol, diff, 1)

  def apply(inputCol: String, timeCol: String):
  TimeSeriesDiff = new TimeSeriesDiff(inputCol, timeCol, 1, 1)
}
