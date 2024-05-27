object ScalaWarmUp{

    def getSecondToLast1(lst: List[Int]): Option[Int] = lst match {
        case Nil | _ :: Nil => None
        case secondLast :: _ :: Nil => Some(secondLast)
        case _ :: tail => getSecondToLast1(tail)
    }
  
    def getSecondToLast2(l: List[Int]): Int = { 
        l.zipWithIndex.filter(_._2 == l.length - 2).head._1
    }
  
    def filterUnique(l: List[String]): List[String] = l match {
        case Nil => Nil
        case head :: tail => head :: filterUnique(tail.filter(_ != head))
    }
  
    def getMostFrequentSubstring(lst: List[String], k: Int) = {
        lst
        .flatMap(_.sliding(k))
        .groupBy(identity)
        .view.mapValues(_.size)
        .toMap
        .maxBy(_._2)
        ._1
    }
    
	//def specialMonotonic(low: Int, high: Int): List[Int] = {
	//	(low to high).filter { num =>
	//	val digits = num.toString.map(_.asDigit)
	//	digits.sliding(2).forall { case Seq(a, b) => math.abs(a - b) == 1 }
	//	}.toList
	//}

    def main(args: Array[String]) = {
        val lst1 = List(5, 3, 7, 2, 8, 1, 9, 4, 6, 0)
        val lst2 = List("a", "b", "c", "a", "b", "c", "a", "b", "c")
        val lst3 = List("abcd", "xwyuzfs", "klmbco")

        println("getSecondToLast1: " + getSecondToLast1(lst1).getOrElse("No second to last element"))
        println("getSecondToLast2: " + getSecondToLast2(lst1))
        println("filterUnique: " + filterUnique(lst2).mkString(", "))
        println("getMostFrequentSubstring: " + getMostFrequentSubstring(lst3, 2))
        //println("specialMonotonic: " + specialMonotonic(100, 200))
    }
}
