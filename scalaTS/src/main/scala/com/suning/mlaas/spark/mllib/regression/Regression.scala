package com.suning.mlaas.spark.mllib.regression

import com.suning.mlaas.spark.mllib.transform.{Feature2Vector, MultiTransformer, String2Index}
import com.suning.mlaas.spark.mllib.util.{Identifiable, Load, Model}
import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{DataType, StringType}

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

abstract class Regression(override val uid: String, features: Array[String], label: String)
  extends Model {
  def this(features: Array[String], label: String) =
    this(Identifiable.randomUID("Regression"), features, label)

  protected var f2v: Feature2Vector = _

  protected var mt: MultiTransformer = _

  var indexedFeatures = List[String]()

  override def fit(df: DataFrame): this.type = {
    val schema = df.schema
    var transformers = List[String2Index]()
    val convertedFeatures = features.map( f => {
          schema(f).dataType match {
            case _: StringType => {
              val indexed = f + "_index"
              indexedFeatures ::= indexed
              transformers ::= String2Index(f, indexed)
              indexed
            }
            case _: DataType  => {
              f
            }
          }
    })

    mt = MultiTransformer(transformers.toArray)
    mt.fit(df)

    f2v = Feature2Vector(convertedFeatures, label)

    fitImpl(f2v.transform(mt.transform(df)))
  }

  protected def fitImpl(df: DataFrame): this.type

  override def transform(df: DataFrame): DataFrame = {
    require(mt != null, "Model is not trained!")
    val schema = df.schema
    val convertedFeatures = features.map( f => {
      schema(f).dataType match {
        case _: StringType => {
          f + "_index"
        }
        case _: DataType  => {
          f
        }
      }
    })
    if (schema.fieldNames.contains(label)) {
      f2v = Feature2Vector(convertedFeatures, label)
    }
    else {
      f2v = Feature2Vector(convertedFeatures)
    }
    val rawDf = f2v.transform(mt.transform(df))
    transformImpl(indexedFeatures.foldLeft(rawDf)((d, f) => {
      d.drop(f)
    }))
  }

  protected def transformImpl(df: DataFrame): DataFrame

  override def copy(): Model

  override def save(path: String): Unit

  override def saveHDFS(sc: SparkContext, path: String): Unit
}

object Regression extends Load[Regression]{

}
