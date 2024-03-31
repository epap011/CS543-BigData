import scala.collection.mutable.ListBuffer
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.linalg.{Vectors => MLVectors}
import org.apache.spark.ml.feature.{PolynomialExpansion, LabeledPoint => MLabeledPoint}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.Pipeline
import breeze.linalg.DenseVector

object MillionSongPipeline {

    private val averageYear = -1

    def main(args: Array[String]): Unit = {
        // --------------------------------------------------------- //
        // ----------------< E X E R C I S E     1 >---------------- //
        // --------------------------------------------------------- //
        val dataset = getClass.getClassLoader.getResource("dataset.csv")

        val spark: SparkSession = SparkSession
            .builder()
            .appName("MillionSongPipeline")
            .master("local[*]")
            .getOrCreate()

        val sc: SparkContext = spark.sparkContext

        val baseRdd = sc.textFile(dataset.getPath)

        val numOfDataPoints = baseRdd.count()
        println("[1.2.1] Number of data points: " + numOfDataPoints) //1.2.1

        println("[1.2.2] Top Five data points")
        baseRdd.take(5).foreach(println) //1.2.2

        val parsedPointsRdd = baseRdd.map(stringToLabeledPoint) //1.3.2

        println("label of the first element of parsedPointsRdd: " + parsedPointsRdd.first().label) //1.3.3
        println("features of the first element of parsedPointsRdd: " + parsedPointsRdd.first().features) //1.3.4
        println("length of the features of the first element of parsedPointsRdd: " + parsedPointsRdd.first().features.size) //1.3.5

        val smallestLabel = parsedPointsRdd.map(_.label).min()
        println("(parsed data) the smallest label: " + smallestLabel) //1.3.6
        val largestLabel = parsedPointsRdd.map(_.label).max()
        println("(parsed data) the largest label: " + largestLabel)  //1.3.7

        val shiftedPointsRdd = parsedPointsRdd.map(lp => LabeledPoint(lp.label - smallestLabel, lp.features)) //1.4.1
        println("(shifted data) the smallest label: " + shiftedPointsRdd.map(_.label).min()) //1.4.2
        println("(shifted data) the largest label: " + shiftedPointsRdd.map(_.label).max())  //1.4.2

        //1.5.1 | Training, validation and test sets
        val weights = Array(.8, .1, .1)
        val seed = 42
        val Array(trainData, valData, testData) = shiftedPointsRdd.randomSplit(weights, seed)

        //1.5.2
        trainData.cache()
        valData.cache()
        testData.cache()

        //1.5.3
        val trainDataCount = trainData.count()
        val valDataCount   = valData.count()
        val testDataCount  = testData.count()
        val sumCounts = trainDataCount + valDataCount + testDataCount

        println("Number of elements in trainData: " + trainDataCount)
        println("Number of elements in valData: " + valDataCount)
        println("Number of elements in testData: " + testDataCount)
        println("Sum of counts: " + sumCounts)

        val totalCountShifted = shiftedPointsRdd.count()
        println("Total number of elements in shiftedPointsRdd: " + totalCountShifted)
        println("Is the sum of counts equal to the count of shiftedPointsRdd? " + (sumCounts == totalCountShifted))

        // --------------------------------------------------------- //
        // ----------------< E X E R C I S E     2 >---------------- //
        // --------------------------------------------------------- //

        //2.1.1
        val averageYear = trainData.map(_.label).mean()
        println("Average (shifted) song year on the training set: " + averageYear)

        val predsTrain = trainData.map(baseLineModel)
        val predsVal   = valData.map(baseLineModel)
        val predsTest  = testData.map(baseLineModel)

        val predsNLabelsTrain = predsTrain.zip(trainData.map(_.label))
        val predsNLabelsVal   = predsVal.zip(valData.map(_.label))
        val predsNLabelsTest  = predsTest.zip(testData.map(_.label))

        println("RMSE on training set: " + calcRmse(predsNLabelsTrain))
        println("RMSE on validation set: " + calcRmse(predsNLabelsVal))
        println("RMSE on test set: " + calcRmse(predsNLabelsTest))

        // --------------------------------------------------------- //
        // ----------------< E X E R C I S E     3 >---------------- //
        // --------------------------------------------------------- //

        //3.3
        //val exampleN = 4
        //val exampleD = 3
        //val exampleData = sc.parallelize(trainData.take(exampleN)).map(lp => LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.slice(0, exampleD))))
        val exampleNumIters = 50
        val (exampleWeights, exampleErrorTrain) = lrgd(trainData, exampleNumIters)
        println("alpha: " + 0.001 + "\nIterations: " + exampleNumIters + "\nTotalErrors(per iteration): " + exampleErrorTrain + "\nModel Parameters: " + exampleWeights)
        println(exampleErrorTrain)
        // 3.4
        val predsNLabelsVal_2 = valData.map(lp => getLabeledPrediction(exampleWeights, lp))
        val rmseVal = calcRmse(predsNLabelsVal_2)
        println("RMSE on the validation set: " + rmseVal)

        //Note 0: Num of iterations are not responsible, because the error always increases towards the infinity, so the gradient
        //        descent does not converge!
        //Note 1: Since the num of iterations is not the case, it suggests that learning step is too large.
        //        So the algorithm overshoots the minimum and diverge. There is a need for different alphas, where a=0.001 seems ok (a=0.01 still overshoots)

        // --------------------------------------------------------- //
        // ----------------< E X E R C I S E     4 >---------------- //
        // --------------------------------------------------------- //

        // 4.1 - MLlib Linear Regression
        /*********************RDD conversion to Dataframe*****************/
        import spark.implicits._
        val trainDataDF = trainData.map(lp => MLabeledPoint(lp.label, MLVectors.dense(lp.features.toArray))).toDF
        val valDataDF   = valData.map(lp => MLabeledPoint(lp.label, MLVectors.dense(lp.features.toArray))).toDF
        val testDataDF  = testData.map(lp => MLabeledPoint(lp.label, MLVectors.dense(lp.features.toArray))).toDF

        /******Linear Regression Demo*********/
        val lr = new LinearRegression()
            .setMaxIter(50)
            .setRegParam(0.1)
            .setFitIntercept(true)

        val lrModel = lr.fit(trainDataDF)
        lrModel.evaluate(valDataDF).rootMeanSquaredError

        // 4.1.1
        println("Coefficients: " + lrModel.coefficients + "\nIntercept: " + lrModel.intercept)

        // 4.1.2
        val rmse = lrModel.evaluate(valDataDF).rootMeanSquaredError
        println("RMSE on the validation set: " + rmse)

        // 4.1.3
        val valPredictions = lrModel.transform(valDataDF)
        println("First 10 predictions:")
        val valPredictionsFirst10 = valPredictions.select("prediction").take(10)
        valPredictionsFirst10.foreach(println)

        // 4.2 - Grid Search
        val regParams = Array(1e-10, 1e-5, 1)

        val paramGrid = new ParamGridBuilder()
            .addGrid(lr.regParam, regParams)
            .build()

        val cv = new CrossValidator()
            .setEstimator(lr)
            .setEvaluator(new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse"))
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(3)

        val cvModel = cv.fit(trainDataDF)
        val bestModel = cvModel.bestModel.asInstanceOf[LinearRegressionModel]
        val rmseBestModel = cvModel.avgMetrics.min

        println("RMSE of the best model: " + rmseBestModel) //4.2.1
        println("Regularization parameter of the best model: " + bestModel.getRegParam) //4.2.2

        // --------------------------------------------------------- //
        // ----------------< E X E R C I S E     5 >---------------- //
        // --------------------------------------------------------- //

        //5.1
        implicit class Crossable[X](xs: Traversable[X]) {
            def cross[Y](ys: Traversable[Y]) = for { x <- xs; y <- ys } yield (x, y)
        }

        def quadFeatures(lp: LabeledPoint) = {
            val crossFeatures = lp.features.toArray.toList cross lp.features.toArray.toList
            val sqFeatures = crossFeatures.map(x => x._1 * x._2).toArray
            MLabeledPoint(lp.label, MLVectors.dense(sqFeatures))
        }

        val crossTrainDataRDD = trainData.map(lp => quadFeatures(lp))

        val crossTrainDataDF: DataFrame = crossTrainDataRDD.toDF("label", "features")
        val crossValDataDF: DataFrame   = valData.map(lp => quadFeatures(lp)).toDF("label", "features")  //5.1.2
        val crossTestDataDF: DataFrame  = testData.map(lp => quadFeatures(lp)).toDF("label", "features") //5.1.2

        //5.2
        val numIters = 500
        val reg = 1e-10
        val useIntercept = true
        
        val lr5 = new LinearRegression()
            .setMaxIter(numIters)
            .setRegParam(reg)
            .setFitIntercept(useIntercept)        

        val lrModel5 = lr5.fit(crossTrainDataDF)

        // 5.3 Evaluate interaction model on validation data
        val rmseVal5 = lrModel5.evaluate(crossValDataDF).rootMeanSquaredError

        // 5.4 Evaluate interaction model on test data
        val predictionsDF = lrModel5.transform(crossTestDataDF)

        val evaluator = new RegressionEvaluator()
            .setLabelCol("label") // Specify the label column
            .setPredictionCol("prediction") // Specify the prediction column
            .setMetricName("rmse") // Specify the evaluation metric (RMSE)
        val rmseTest5 = evaluator.evaluate(predictionsDF)

        println(s"RMSE on validation set: $rmseVal5")
        println(s"RMSE on test set: $rmseTest5")

        predictionsDF.select("label").show(50)

        //5.5
        //val numIters = 500
        //val reg = 1e-10
        val alpha = .2
        //val useIntercept = true
        val polynomial_expansion = (new PolynomialExpansion).setInputCol("features").setOutputCol("polyFeatures").setDegree(2)
        val lr3 = new LinearRegression()
        lr3.setMaxIter(numIters).setRegParam(reg).setElasticNetParam(alpha).setFitIntercept(useIntercept).setFeaturesCol("polyFeatures")

        val pipeline = new Pipeline()
        pipeline.setStages(Array(polynomial_expansion, lr)) //there are two stages here that you have to set.

        val model=pipeline.fit(trainDataDF) //need to fit. Use the train Dataframe
        val predictionsDF5=model.transform(testDataDF) //Produce predictions on the test set. Use method transform.
        val evaluator5 = new RegressionEvaluator()
        evaluator5.setMetricName("rmse")
        val rmseTestPipeline = evaluator5.evaluate(predictionsDF5)
        println(rmseTestPipeline)
    }

    //1.3.1
    private def stringToLabeledPoint(data: String): LabeledPoint = {
        val values   = data.split(",").map(_.toDouble)
        val label    = values.head
        val features = Vectors.dense(values.tail)
        LabeledPoint(label, features)
    }

    //2.1.2
    private def baseLineModel(lp: LabeledPoint): Double = {
        averageYear
    }

    //2.2
    private def calcRmse(predictionsAndLabels: RDD[(Double, Double)]): Double = {
        val metrics = new RegressionMetrics(predictionsAndLabels)
        metrics.rootMeanSquaredError
    }

    //3.1
    private def gradientSummand(weights: DenseVector[Double], lp: LabeledPoint): DenseVector[Double] = {
        val featuresBreeze = DenseVector(lp.features.toArray)
        val prediction = weights.dot(featuresBreeze)
        val error = prediction - lp.label
        val gradientSummand = featuresBreeze * error
        gradientSummand
    }

    //3.2
    private def getLabeledPrediction(weights: DenseVector[Double], lp: LabeledPoint): (Double, Double) = {
        val featuresBreeze = DenseVector(lp.features.toArray)
        val prediction = weights.dot(featuresBreeze)
        (prediction, lp.label)
    }

    //3.3.1
    private def lrgd(trData: RDD[LabeledPoint], numIter: Int): (DenseVector[Double], List[Double]) = {
        val n = trData.count
        val d = trData.first.features.size
        val alpha = 0.002
        val errorTrain = new ListBuffer[Double]
        var weights = new DenseVector(Array.fill[Double](d)(0.0))
        for (i <- 0 until numIter){
            val gradient = trData.map { lp =>
                val summand = gradientSummand(weights, lp)
                summand
            }.reduce(_ + _)
            val alpha_i = alpha / (n * Math.sqrt(i+1))
            weights -= gradient * alpha_i
            //update errorTrain
            val predsNLabelsTrain = trData.map(lp => getLabeledPrediction(weights, lp))
            errorTrain += calcRmse(predsNLabelsTrain)
        }
        (weights, errorTrain.toList)
    }
}