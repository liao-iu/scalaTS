package com.suning.mlaas.spark.mllib.SQLData

import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame,Row}

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

object ToRDD {

  def toRDDVector(df:DataFrame):RDD[Vector] = {
    val dataResult = df.rdd.map(row => {
      Vectors.dense(row.toSeq.toArray.map(
        {
          case s: String => s.toDouble
          case l: Long => l.toDouble
          case d: Double => d.toDouble
          case i: Int => i.toDouble
          case f: Float => f.toDouble
          case b:Byte => b.toDouble
          case _ => 0.0
        }
      ))
    })
    dataResult
  }

  def toRddRow(df:DataFrame):RDD[Row] = {

    val dataResult = df.rdd.map(row => {
      Row.fromSeq(row.toSeq.toArray.map(
        {
          case s: String => s.toDouble
          case l: Long => l.toDouble
          case d: Double => d.toDouble
          case i: Int => i.toDouble
          case f: Float => f.toDouble
          case b: Byte => b.toDouble
          case _ => 0.0
        }
      ))
    })
    dataResult

  }

}
