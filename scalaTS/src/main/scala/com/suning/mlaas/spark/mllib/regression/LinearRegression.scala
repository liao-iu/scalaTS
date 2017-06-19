package com.suning.mlaas.spark.mllib.regression

import com.suning.mlaas.spark.mllib.util.Model
import com.suning.mlaas.spark.mllib.util.{Identifiable, SaveLoad}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.{LinearRegression => SparkLR, LinearRegressionModel => SparkLRModel}
import org.apache.spark.sql.DataFrame

/**
  *  Copyright [2016/7] [Ao Li @ Suning R&D]
  */

class LinearRegression(override val uid: String, features: Array[String], label: String,
                       regParam: Double,
                       fitIntercept: Boolean,
                       standardization: Boolean,
                       elasticNetParam: Double,
                       maxIter: Int,
                       tol: Double)
  extends Regression(uid, features, label) {

  def this(features: Array[String], label: String,
           regParam: Double = 0.0,
           fitIntercept: Boolean = true,
           standardization: Boolean = true,
           elasticNetParam: Double = 0.0,
           maxIter: Int = 100,
           tol: Double = 1E-6) =
    this(Identifiable.randomUID("LinearRegression"), features, label, regParam,
      fitIntercept, standardization, elasticNetParam, maxIter, tol)

  private var model: SparkLRModel = _

  private val lr = new SparkLR().setRegParam(regParam)
    .setFitIntercept(fitIntercept)
    .setStandardization(standardization)
    .setElasticNetParam(elasticNetParam)
    .setMaxIter(maxIter)
    .setTol(tol)

  protected def fitImpl(df: DataFrame): this.type = {
    model = lr.fit(df)
    this
  }

  protected def transformImpl(df: DataFrame): DataFrame = {
    model.transform(df)
  }

  override def copy(): Model = {
    new LinearRegression(features, label, regParam, fitIntercept,
      standardization, elasticNetParam, maxIter, tol)
  }

  def getIntercept(): Double = {
    model.intercept
  }

  def getWeights(): Vector = {
    model.coefficients
  }

  override def save(path: String): Unit = {
    LinearRegression.save(this, path)
  }

  override def saveHDFS(sc: SparkContext, path: String): Unit = {
    LinearRegression.saveHDFS(sc, this, path)
  }
}

object LinearRegression extends SaveLoad[LinearRegression]{
  def apply(features: Array[String], label: String):
  LinearRegression = new LinearRegression(features, label)

  def apply(features: Array[String], label: String, regParam: Double,
            fitIntercept: Boolean,
            standardization: Boolean,
            elasticNetParam: Double,
            maxIter: Int,
            tol: Double):
  LinearRegression = new LinearRegression(features, label, regParam,
    fitIntercept, standardization, elasticNetParam, maxIter, tol)

  def apply(uid: String, features: Array[String], label: String, regParam: Double,
            fitIntercept: Boolean,
            standardization: Boolean,
            elasticNetParam: Double,
            maxIter: Int,
            tol: Double):
  LinearRegression = new LinearRegression(uid, features, label, regParam,
    fitIntercept, standardization, elasticNetParam, maxIter, tol)

}
