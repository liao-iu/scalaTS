package com.suning.mlaas.spark.mllib

import org.apache.log4j.{Level, LogManager, PropertyConfigurator}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

trait SparkSuite extends FunSuite with BeforeAndAfterAll {

  protected var sparkContext: SparkContext = _
  protected var sparkSession: SparkSession = _
  protected var df: DataFrame = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    PropertyConfigurator.configure("log4j.properties")
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val conf = new SparkConf()
    conf.setMaster("local[4]")
    conf.setAppName(s"${this.getClass.getSimpleName}Suite")
    //    sparkContext = new SparkContext(conf)
    val warehouseLocation = "file:${system:user.dir}/spark-warehouse"
    sparkSession = SparkSession
      .builder()
      .config(conf)
      //      .config("spark.sql.warehouse.dir", warehouseLocation)
      //      .enableHiveSupport()
      .getOrCreate()
    sparkContext = sparkSession.sparkContext
  }

  override def afterAll(): Unit = {
    try {
      sparkSession.stop()
    } finally {
      super.afterAll()
    }
  }
}
