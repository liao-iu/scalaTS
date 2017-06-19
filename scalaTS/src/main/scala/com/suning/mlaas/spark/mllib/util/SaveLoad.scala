package com.suning.mlaas.spark.mllib.util

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream, FileSystem, Path}
import org.apache.spark.SparkContext

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
trait SaveLoad[T] extends Save[T] with Load[T]{
}
