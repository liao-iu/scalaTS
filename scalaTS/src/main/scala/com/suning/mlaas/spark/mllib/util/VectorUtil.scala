package com.suning.mlaas.spark.mllib.util

import org.apache.spark.sql.types.{DataType, StringType}
import org.apache.spark.mllib.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

object VectorUtil {

  val s2v = udf((string: String) => {
    Vectors.parse(string)
  })

  val v2s = udf((features: Vector) => {
    features.toString
  })

  def string2Vector(df: DataFrame): DataFrame = {
    val schema = df.schema
    var newDF = df
    schema.foreach( field => {
      field.dataType match {
        case _: StringType => {
          val colName = field.name
          if (colName.startsWith("v2s_")) {
            val newColName = colName.replace("v2s_", "")
            newDF = newDF.withColumn(newColName, s2v(newDF(colName))).drop(newDF(colName))
          }
        }
        case _: DataType => {
        }
      }
    })
    newDF
  }

  def vector2String(df: DataFrame): DataFrame = {
    val schema = df.schema
    var newDF = df
    schema.foreach( field => {
      field.dataType match {
        case _: VectorUDT => {
          val colName = field.name
          val newColName = "v2s_" + field.name
          newDF = newDF.withColumn(newColName, v2s(newDF(colName))).drop(newDF(colName))
        }
        case _: DataType => {
        }
      }
    })
    newDF
  }
}
