## Welcome to scalaTS GitHub Page
### Our distributed Time Series Analysis with DataFrames

Docs of Updates are published here and the package is hosted at this [GitHub page](https://github.com/liao-iu/scalaTS/tree/master/scalaTS). This package is introduced and present at the Spark Summit 2017, San Francisco, [slides](https://www.slideshare.net/databricks/machine-learning-as-a-service-apache-spark-mllib-enrichment-and-webbased-codeless-modeling-with-zhengyi-le).

### Content

-	Time series analysis with DataFrames of Apache Spark
-	Experiments
-	What’s in the package - Time Series Utilities and Models

### Time series analysis with DataFrames of Apache Spark

Time series analysis is widely used in predictive models. Now, big data technologies dramatically decrease searching and mining time and make analysis more accurate. However, it also brings new challenges/demands to big TS data modeling in order to make TS analysis more efficient. The challenges include:
- (1) distribute time series algorithms to incorporate more massive historical data;
- (2) support complex data format/schema to enable advanced data processing;
- (3) incorporate them into most popular large-scale data processing engine to obtain an integrated solution for data access, processing and modeling.

This package introduces our distributed time series algorithm family: scalable Time Series (scalaTS) on Spark DataFrames to address the above challenges. scalaTS is faster (on average) and more powerful (supporting more complex data format and algorithms) than existing open source time series algorithms and packages.

The scalaTS analysis on Spark aims at building time series analysis for distributed data sets based on Scala/ Java. The utilities and models of scalaTS are directly implemented with Spark's [DataFrames](http://spark.apache.org/docs/latest/sql-programming-guide.html#dataframe-operations). Compared to traditional RDD data structure, DataFrame brings Apache Spark the following new features:
- support data size from KB to PB.
- support variant data types and storage systems.
- support SQL-like DSL(domain specific language) and all 99 TPC-DS queries.
- generate efficient code by Catalyst Optimizer in Spark SQL.
- seamless integrated with other big data toolkits in Spark and other infrastructures.
- provide APIs in scala, java python and R.
From Spark 1.6, DataFrame is evolved into [DataSet](https://databricks.com/blog/2016/01/04/introducing-apache-spark-datasets.html) which supports more user-defined data type rather than Row.

Another popular open source time series package on Spark ([Spark-ts](http://sryza.github.io/spark-timeseries/0.3.0/index.html)). The author considered three data structures: two are DataFrame based - observations and instants; the third is RDD based, named [TimeSeriesRDD](https://blog.cloudera.com/blog/2015/12/spark-ts-a-new-library-for-analyzing-time-series-data-with-apache-spark/). The TimeSeriesRDD[K] extends RDD[(K, Vector[Double])], where K is the key type (usually a String), and the second element in the tuple is a Breeze vector representing the time series. It only parallelizes every series, not within a series which is still a breeze vector. Not like a breeze vector, our scalaTS takes fully advantage of DataFrames and also enriches Spark ML with time series analysis.

### Experiments

The scalaTS package keeps enabling time series utilities and models for large-scale time series data sets, as analogous as R's [forecast package](https://cran.r-project.org/web/packages/forecast), Matlab's [time series](http://www.mathworks.com/help/matlab/time-series.html), [Spark-ts](http://sryza.github.io/spark-timeseries/0.3.0/index.html)...

![Running time](https://github.com/liao-iu/scalaTS/raw/master/docs/images/runningTime.png)

### What’s in the package - Time Series Utilities and Models
#### Time Series Utilities
* Time series decomposition:

It is an important technique for every time series analysis. It decomposes an observed time series into several components including predictable and unpredictable components. Both [additive and multiplicative time series decomposition](https://en.wikipedia.org/wiki/Decomposition_of_time_series) are available.
* Time series lagging:

Time series lagging generates DataFrames of Spark with lagged terms given an observed time series.
* Time series differencing:

Time series differencing gives DataFrames of Spark with differenced terms given an observed time series.
* Save or Load models:

The scalaTS supports both saving and loading time series models.

#### Time Series Models
* Autocorrelation function:

The [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation) reflects the correlation between values of a given time series and at different time lags. It looks for repeating patterns to analyze functions or series of values. For the estimation of a moving average model, the autocorrelation function is applied to determine a proper number of lagged terms (denoted as q).
* Partial autocorrelation function (implement both Yule-Walker and OLS methods):

With controlling the smaller lags of time series, the [partial autocorrelation function](https://en.wikipedia.org/wiki/Partial_autocorrelation_function) computes the partial correlation with its lagged values. For an autoregressive model, the partial autocorrelation function features identifying the length of lags and deciding an appropriate lags (denoted as p).
* AIC/ AICc/ BIC calculations:

The scalaTS enables calculations of information criterions including Akaike information criterions ([AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion))/ AIC with correction for finite sample sizes ([AICc](https://en.wikipedia.org/wiki/Akaike_information_criterion))/ Bayesian information criterion ([BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion)).
* Autoregressive regression (AR) (implement both Yule-Walker and OLS methods):

In statistics and signal process, the [autoregressive model](https://www.otexts.org/fpp/8/3) fits the values of a time series with its own lagged terms and a stochastic term.
* Auto-AR model:

The auto autoregressive model fits an AR model with a given range of lags (p) according to different information criterions including AIC/ AICc/ BIC. It could automatically choose the best number of lags for AR model.
* Moving average (MA) regression:

Other than AR model's using lagged terms for a regression, the [moving average model](https://www.otexts.org/fpp/8/4) exploits past forecasting errors in the regression model.
* Auto-MA model:

The auto moving average model fits an MA model with a given range of values (q) according to different information criterions including AIC/ AICc/ BIC. It could automatically choose the best number of q for MA model.
* Autoregressive moving average (ARMA) regression:

Given a time series, the [ARMA](https://en.wikipedia.org/wiki/Autoregressive-moving-average_model) plays an important role in understanding and predicting future values in the series. The model combines both AR model and MA model. The AR part fits the values of the series on its own lagged values and the MA part linearly models the error term with the past errors.
* Auto-ARMA model:

The auto ARMA model finds best order of the autoregressive part (p) and order of the moving average part (q) according to different information criterions including AIC/ AICc/ BIC. It could automatically choose the best number of (p, q) for ARMA model.
* Autoregressive integrated moving average (ARIMA) regression:

An [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) model is an generalization of the ARMA model. The difference between ARMA model and ARIMA model is that the time series values in the ARIMA model are replaced by the differencing values. And the times of differencing are decided by the order (denoted as d).
* Auto-ARIMA model:

The auto ARIMA model picks best order of the autoregressive part (p), order of the moving average part (q) and order of the integrated part (d) according to different information criterions including AIC/ AICc/ BIC. It could automatically choose the best number of (p, d, q) for ARIMA model.


### Sponsor and Supporting Team
The authors, Ao Li and Jin Xu, finished this scalaTS project during the time at Big Data Lab of [Suning R&D Palo Alto](http://www.ussuning.com/). Thanks are due to the team members for their supports: Kuangyu Wang, Ming Jiang, Weizhi Li, Xueting Shao, Ji Dai and Zhengyi Le.

![logo](https://github.com/liao-iu/scalaTS/raw/master/docs/images/Suning_word.png)

Copyright [2016/7] [Big Data lab, Suning R&D]

© 2016 Big Data lab, Suning, USA.
