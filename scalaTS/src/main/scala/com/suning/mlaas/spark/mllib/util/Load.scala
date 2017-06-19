package com.suning.mlaas.spark.mllib.util

import java.io.{FileInputStream, ObjectInputStream}
import org.apache.hadoop.fs.{FSDataInputStream, FileSystem, Path}
import org.apache.spark.SparkContext

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
trait Load[T] {

  def load(path: String): T = {
    val is = new ObjectInputStream(new FileInputStream(path))
    val t = is.readObject().asInstanceOf[T]
    is.close()
    t
  }

  def loadHDFS(sc: SparkContext, path: String): T = {
    val hadoopConf = sc.hadoopConfiguration
    val fileSystem = FileSystem.get(hadoopConf)
    val HDFSPath = new Path(path)
    val ois = new ObjectInputStream(new FSDataInputStream(fileSystem.open(HDFSPath)))
    val t = ois.readObject.asInstanceOf[T]
    ois.close()
    t
  }
}
