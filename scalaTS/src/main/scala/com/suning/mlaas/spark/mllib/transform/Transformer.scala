package com.suning.mlaas.spark.mllib.transform

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import com.suning.mlaas.spark.mllib.Logging
import com.suning.mlaas.spark.mllib.util.{Identifiable, Load, SaveLoad}
import org.apache.spark.sql.DataFrame
import org.apache.spark.SparkContext

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */

abstract class Transformer(override val uid: String, keepOriginal: Boolean = true)
  extends Logging with Identifiable with Serializable {

  def this(keepOriginal: Boolean) =
    this(Identifiable.randomUID("Transformer"), keepOriginal)

  def this() =
    this(Identifiable.randomUID("Transformer"), true)

  def transform(df: DataFrame): DataFrame = {
    if (keepOriginal) transformImpl(df) else removeOriginal(transformImpl(df))
  }

  def transformImpl(df: DataFrame): DataFrame

  def fit(df: DataFrame): this.type = {
    fitImpl(df)
  }

  protected def fitImpl(df: DataFrame): this.type

  def removeOriginal(df: DataFrame): DataFrame

  def save(path: String): Unit

  def saveHDFS(sc: SparkContext, path: String): Unit
}


object Transformer extends Load[Transformer]{

}
