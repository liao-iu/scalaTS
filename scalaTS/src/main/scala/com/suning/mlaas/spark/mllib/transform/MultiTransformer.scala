package com.suning.mlaas.spark.mllib.transform

import com.suning.mlaas.spark.mllib.util.{Identifiable, SaveLoad}
import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */

class MultiTransformer(override val uid: String, transformers: Array[Transformer] )
  extends Transformer (uid) {
  def this(transformers: Array[Transformer]) =
    this(Identifiable.randomUID("MultiTransformer"), transformers)

  override def transformImpl(df: DataFrame): DataFrame = {
    transformers.foldLeft(df)((d, t) => t.transform(d))
  }

  override def fitImpl(df: DataFrame): this.type = {
    transformers.foreach(_.fit(df))
    this
  }

  override def removeOriginal(df: DataFrame): DataFrame = {
    df
  }

  override def save(path: String): Unit ={
    MultiTransformer.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    MultiTransformer.saveHDFS(sc, this, path)
  }
}

object MultiTransformer extends SaveLoad[MultiTransformer] {
  def apply(transformers: Array[Transformer]): MultiTransformer =
    new MultiTransformer(transformers)
}
