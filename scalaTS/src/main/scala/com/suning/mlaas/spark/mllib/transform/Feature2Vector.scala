package com.suning.mlaas.spark.mllib.transform

import com.suning.mlaas.spark.mllib.util.{Identifiable, SaveLoad}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{VectorAssembler => SparkVA}
import org.apache.spark.sql.DataFrame

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */

class Feature2Vector(override val uid: String, keepOriginal: Boolean,
                     features: Array[String], label: String)
  extends Transformer(uid, keepOriginal) {

  def this(keepOriginal: Boolean, features: Array[String], label: String) =
    this(Identifiable.randomUID("Feature2Vector"), keepOriginal, features, label)

  def this(features: Array[String], label: String) =
    this(Identifiable.randomUID("Feature2Vector"), true, features, label)

  protected val va = new SparkVA().setInputCols(features).setOutputCol("features")

  override def transformImpl(df: DataFrame): DataFrame = {
    if( label == "") {
      va.transform(df)
    } else {
      va.transform(df).withColumn("label", df(label))
    }
  }

  override def removeOriginal(df: DataFrame): DataFrame = {
    val cols = df.schema.fieldNames.filter(!features.contains(_))
    df.select(cols.head, cols.tail: _*)
  }

  override def fitImpl(df: DataFrame): this.type = {
    this
  }

  override def save(path: String): Unit ={
    Feature2Vector.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    Feature2Vector.saveHDFS(sc, this, path)
  }
}

object Feature2Vector extends SaveLoad[Feature2Vector]{
  def apply(uid: String, keepOriginal: Boolean, features: Array[String], label: String):
  Feature2Vector = new Feature2Vector(uid, keepOriginal, features, label)

  def apply(keepOriginal: Boolean, features: Array[String], label: String):
  Feature2Vector = new Feature2Vector(keepOriginal, features, label)

  def apply(features: Array[String], label: String):
  Feature2Vector = new Feature2Vector(features, label)

  def apply(features: Array[String]):
  Feature2Vector = new Feature2Vector(features, "")
}
