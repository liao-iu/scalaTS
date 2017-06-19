package com.suning.mlaas.spark.mllib

import scala.collection.GenTraversable
import scala.language.implicitConversions

/**
  * Copyright [2016/7] [Big Data lab, Suning R&D]
  */

object ImplicitImports {

  implicit def toPP(any: Any): PP = new PP(any)

  class PP(any: Any) {
    def pretty(): Unit = println(any match {
      case col: GenTraversable[_] => col.mkString("<---\n", "\n", "\n-->")
      case arr: Array[_] => arr.mkString("<---\n", "\n", "\n-->")
      case e => Seq("<---", e, "--->").mkString
    })
  }

}
