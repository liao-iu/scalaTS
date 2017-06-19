package com.suning.mlaas.spark.mllib.util

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}
import org.apache.spark.SparkContext

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */
  
trait Save[T]{
  def save(t: T, path: String): Unit = {
    val os = new ObjectOutputStream(new FileOutputStream(path))
    os.writeObject(t)
    os.close()
  }

  def saveHDFS(sc: SparkContext, t: T, path: String): Unit = {
    val hadoopConf = sc.hadoopConfiguration
    val fileSystem = FileSystem.get(hadoopConf)
    val HDFSPath = new Path(path)
    val oos = new ObjectOutputStream(new FSDataOutputStream(fileSystem.create(HDFSPath), null))
    oos.writeObject(t)
    oos.close()
  }
}
