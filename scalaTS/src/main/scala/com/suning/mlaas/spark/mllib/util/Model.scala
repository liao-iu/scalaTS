package com.suning.mlaas.spark.mllib.util

import com.suning.mlaas.spark.mllib.Logging
import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
abstract class Model(override val uid: String)
  extends Logging with Identifiable with Serializable {
  def this() =
    this(Identifiable.randomUID("Model"))


  def fit(df: DataFrame): this.type = {
    fitImpl(df)
  }

  protected def fitImpl(df: DataFrame): this.type

  def transform(df: DataFrame): DataFrame = {
    transformImpl(df)
  }

  protected def transformImpl(df: DataFrame): DataFrame

  def save(path: String): Unit

  def saveHDFS(sc: SparkContext, path: String): Unit

  def copy(): Model
}

object Model extends Load[Model] {

}
