package com.suning.mlaas.spark.mllib.util

import java.util.UUID

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

trait Identifiable {

  /**
    * An immutable unique ID for the object and its derivatives.
    */
  val uid: String

  override def toString: String = uid
}


object Identifiable {

  /**
    * Returns a random UID that concatenates the given prefix, "_", and 16 random hex chars.
    */
  def randomUID(prefix: String): String = {
    prefix + "_" + UUID.randomUUID().toString.takeRight(16)
  }
}
