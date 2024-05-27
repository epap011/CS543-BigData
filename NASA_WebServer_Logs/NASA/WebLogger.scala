import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

object WebLogger {
    private val patternHost    = """^([^\s]+\s)""".r
    private val patternTime    = """^.*\[(\d\d/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]""".r
    private val patternRequest = """^.*"\w+\s+([^\s]+)\s*.*"""".r
    private val patternStatus  = """^.*"\s+([^\s]+)""".r
    private val patternBytes   = """^.*\s+(\d+)$""".r
    private case class Log(host: String, date: String, requestURI: String, status: Int, bytes: Int)

    private def convertToLog(base: RDD[String]): RDD[Log] = {
        val splitRdd = base.map(getImprovedLogFields)
        //val cleanRdd = splitRdd.filter(log => patternRequest.findFirstIn(log.requestURI) != None)
        splitRdd
    }

    private def getLogFields(str: String):Log = {
        Log (patternHost.findAllIn(str).matchData.next().group(1),
            patternTime.findAllIn(str).matchData.next().group(1),
            patternRequest.findAllIn(str).matchData.next().group(1),
            patternStatus.findAllIn(str).matchData.next().group(1).toInt,
            patternBytes.findAllIn(str).matchData.next().group(1).toInt
        )
    }

    private def getImprovedLogFields(str: String):Log = {
        val host   = patternHost.findFirstMatchIn(str).map(_.group(1)).getOrElse("0")
        val date   = patternTime.findFirstMatchIn(str).map(_.group(1)).getOrElse("0")
        val req    = patternRequest.findFirstMatchIn(str).map(_.group(1)).getOrElse("0")
        val status = patternStatus.findFirstMatchIn(str).map(_.group(1)).getOrElse("0").toInt
        val bytes  = patternBytes.findFirstMatchIn(str).map(_.group(1)).getOrElse("0").toInt

        Log(host, date, req, status, bytes)
    }

    def main(args: Array[String]): Unit = {
        val dataset = getClass.getClassLoader.getResource("NASA_access_log_Jul95")

        val sc = new SparkContext("local[*]", "WebLogger")
        val baseRdd = sc.textFile(dataset.getPath)
        val cleanRdd = convertToLog(baseRdd)

        //min, max, and average content size
        val contentSize = cleanRdd.map(log => log.bytes)
        val minSize = contentSize.min()
        val maxSize = contentSize.max()
        val avgSize = contentSize.mean()
        println(s"Min: $minSize, Max: $maxSize, Avg: $avgSize")

        //100 most frequent status values and their frequency
        val statusCount = cleanRdd.map(log => (log.status, 1)).reduceByKey(_ + _).cache()
        val sortedStatus = statusCount.sortBy(_._2)
        println("Status Count")
        statusCount.take(100).foreach(println)

        //10 hosts that accessed the server more than 10 times.
        val hostCount = cleanRdd.map(log => (log.host, 1)).reduceByKey(_ + _).cache()
        val sortedHost = hostCount.sortBy(_._2, ascending = false)
        println("Host Count")
        sortedHost.take(10).foreach(println)

        //the top 10 requestURIs that did not have a return code of 200.
        val errorPaths = cleanRdd.filter(log => log.status != 200).map(log => (log.requestURI, 1)).reduceByKey(_ + _).cache()
        val sortedErrorPaths = errorPaths.sortBy(_._2, ascending = false)
        println("Error Paths")
        sortedErrorPaths.take(10).foreach(println)

        //how many unique there are in the entire log.
        val uniqueHosts = cleanRdd.map(log => log.host).distinct().count()
        println("Unique Hosts: " + uniqueHosts)

        //the count of 404 Response codes
        val notFound = cleanRdd.filter(log => log.status == 404).count()

        //40 distinct requestURIs that generate 404 errors.
        val notFoundPaths = cleanRdd.filter(log => log.status == 404).map(log => (log.requestURI, 1)).reduceByKey(_ + _).cache()
        println("404 Paths (40 distinct)")
        sortedNotFoundPaths.take(40).foreach(println)

        //a list of the top 20 paths (in sorted order) that generate the most 404 errors.
        val sortedNotFoundPaths2 = notFoundPaths.sortBy(_._2, ascending = false)
        println("404 Paths (in sorted order)")
        sortedNotFoundPaths2.take(20).foreach(println)
    }
}
