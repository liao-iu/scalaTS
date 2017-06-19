package com.suning.mlaas.spark.mllib.transform

import com.suning.mlaas.spark.mllib.util.{Identifiable, SaveLoad}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{StringIndexer => SparkStringIndexer}
import org.apache.spark.ml.feature.{StringIndexerModel => SparkStringIndexerModel}
import org.apache.spark.sql.DataFrame

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */

class String2Index(override val uid: String, keepOriginal: Boolean,
                   inputCol: String, outputCol: String)
  extends Transformer(uid, keepOriginal) {

  def this(keepOriginal: Boolean, inputCol: String, outpuCol: String) =
    this(Identifiable.randomUID("String2Index"), keepOriginal, inputCol, outpuCol)

  def this(inputCol: String, outputCol: String) =
    this(true, inputCol, outputCol)

  private val si = new SparkStringIndexer()
    .setInputCol(inputCol).setOutputCol(outputCol)
  private var siModel: SparkStringIndexerModel = _

  var labels: Array[String] = _

  override def transformImpl(df: DataFrame): DataFrame = {
    if( siModel == null){
      fit(df)
    }
    siModel.transform(df)
  }

  override def removeOriginal(df: DataFrame): DataFrame = {
    df.drop(inputCol)
  }

  override def fitImpl(df: DataFrame): this.type = {
    siModel = si.fit(df)
    labels = siModel.labels
    this
  }

  override def save(path: String): Unit ={
    String2Index.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    String2Index.saveHDFS(sc, this, path)
  }

}

object String2Index extends SaveLoad[String2Index] {
  def apply(uid: String, keepOriginal: Boolean, inputCol: String, outputCol: String):
  String2Index = new String2Index(uid, keepOriginal, inputCol, outputCol)

  def apply(keepOriginal: Boolean, inputCol: String, outputCol: String):
  String2Index = new String2Index(keepOriginal, inputCol, outputCol)

  def apply(inputCol: String, outputCol: String):
  String2Index = new String2Index(inputCol, outputCol)
}
