package com.suning.mlaas.spark.mllib.ts

import com.suning.mlaas.spark.mllib.util.{Identifiable, Load, Model}
import org.apache.spark.sql.DataFrame

/**
  *    Copyright [2016/7] [Ao Li @ Suning R&D]
  */

abstract class TSModel(override val uid: String, inputCol: String, timeCol: String)
  extends Model {
  def this(inputCol: String, timeCol: String) =
    this(Identifiable.randomUID("TSModel"), inputCol, timeCol)

  def forecast(df: DataFrame, numAhead: Int): List[Double]
}

object TSModel extends Load[TSModel] {

}
