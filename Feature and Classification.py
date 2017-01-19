from pyspark import SparkContext, SparkConf
from pyspark.sql import Row
from datetime import datetime
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Definitions
def toIntSafe(inval):
    try:
        return int(inval)
    except ValueError:
        return None

def toTimeSafe(inval):
    try:
        return datetime.strptime(inval, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None

def toDateSafe(inval):
    try:
        return datetime.strptime(inval, "%m/%d/%Y")
    except ValueError:
        return None
    
def toFloatSafe(inval):
    try:
        return float(inval)
    except ValueError:
        return None
        
def toDateTimeSafe(inval):
    try:
        return datetime.strptime(inval, "%m/%d/%Y %H:%M")
    except ValueError:
        return None
        
def convTripString(r):
    return Row(
    toIntSafe(r[0])
    ,toIntSafe(r[1])
    ,toDateTimeSafe(r[2])
    ,r[3]
    ,toIntSafe(r[4])
    ,toDateTimeSafe(r[5])
    ,r[6]
    ,toIntSafe(r[7])
    ,toIntSafe(r[8])
    ,r[9]
    ,r[10]
    )
    
  
def importAndConvert (file_name, conv_function, schema_name, ):
# ### Import YYYYMM_station_data.csv and convert to Data Frame

    # Step 1: Import and remove header for August Data
    data = sc.textFile(file_name)
    header = data.first() #extract header
    data = data.filter(lambda row: row != header) #filter to exclude header
    print "Step 1 complete"
    
    # Step 2: Split rows by commas and convert to tuples
    data_split = data.map(lambda x: x.split(","))
    data_tuple = data_split.map(lambda l: tuple(l))
    print "Step 2 complete"

    # Check: Print out tuple-formatted data
    print data_tuple.take(5)

    # Step 3: Convert Input Data
    data_RDD = data_tuple.map(lambda x: conv_function(x))
    print "Step 3 complete"

    # Step 4: Create DataFrame with Schema
    data_DF = sqlContext.createDataFrame(data_RDD, schema_name)
    print "Step 4 complete"

    # Check: Print Schema
    data_DF.printSchema()

    # Check: Print First 20 Rows of Data Frame
    data_DF.show()

    return data_DF


def stackData(prefix):
	return prefix_201408_DF.unionAll(prefix_201402_DF)
	
def indexStringColumns(df, cols):
    from pyspark.ml.feature import StringIndexer
    #variable newdf will be updated several times
    newdf = df
    for c in cols:
        si = StringIndexer(inputCol=c, outputCol=c+"-num")
        sm = si.fit(newdf)
        newdf = sm.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-num", c)
    return newdf
    
    
# Defining Schemas

tripSchema = StructType([
  StructField("trip_id", IntegerType(), True),
  StructField("duration", IntegerType(), True),
  StructField("start_date", DateType(), True),
  StructField("start_station", StringType(), True),
  StructField("start_terminal", IntegerType(), True),
  StructField("end_date", DateType(), True),
  StructField("end_station", StringType(), True),
  StructField("end_terminal", IntegerType(), True),
  StructField("bike_no", IntegerType(), True),
  StructField("sub_type", StringType(), True),
  StructField("zip", StringType(), True)
  ])
	
# Trip Data
trip_201408_DF = importAndConvert('data/data/201408_trip_data.csv', convTripString, tripSchema)
trip_201402_DF = importAndConvert('data/data/201402_trip_data.csv', convTripString, tripSchema)

trip_12mos = trip_201408_DF.unionAll(trip_201402_DF)

print "Dataset 1 Count: ", trip_201408_DF.count()
print "Dataset 2 Count: ", trip_201402_DF.count()
print "Full 12 MOnths Dataset Count: ", trip_12mos.count()

#trip_201402_DF.show()
trip_12mos.show()

#Number of rides per user
trips_WithRideDuration = trip_12mos.withColumn('duration',datediff('end_date','start_date'))#.sort('duration', ascending = 0)

trips_WithRideDuration.take(5)

def indexStringColumns(df, cols):
    from pyspark.ml.feature import StringIndexer
    #variable newdf will be updated several times
    newdf = df
    for c in cols:
        si = StringIndexer(inputCol=c, outputCol=c+"-num")
        sm = si.fit(newdf)
        newdf = sm.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c+"-num", c)
    return newdf

trip_numeric = indexStringColumns(trips_WithRideDuration, ["sub_type", "start_station", "end_station"])
trip_numeric.show()


# Create feature vector

input_cols=["trip_id","duration","start_date","start_terminal",
            "end_date","end_terminal","bike_no","zip", "start_station", "end_station"]
va = VectorAssembler(outputCol="features", inputCols=input_cols)
trip_points = va.transform(trip_numeric).select("features", "sub_type").withColumnRenamed("sub_type", "label")

# Create Training and Test data.
triptsets = trip_points.randomSplit([0.8, 0.2])
trip_train = triptsets[0].cache()
trip_valid = triptsets[1].cache()



#Random Forest
rf = RandomForestClassifier(maxDepth=10)
rfmodel = rf.fit(trip_train)

rfpredicts = rfmodel.transform(trip_valid)
rfresrdd = rfpredicts.select("prediction", "label").rdd
rfmm = MulticlassMetrics(rfresrdd)
rfmm.precision()

#Decision Tree
dt = DecisionTreeClassifier(maxDepth=20)
dtmodel = dt.fit(trip_train)

dtpredicts = dtmodel.transform(trip_valid)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(dtpredicts)
print("Test Error = %g" % (1.0 - accuracy))

dtresrdd = dtpredicts.select("prediction", "label").rdd #convert DataFrame to RDD.
dtmm = MulticlassMetrics(dtresrdd) 
dtmm.precision()

#Logistic Regression
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(trip_train)

validpredicts = lrmodel.transform(trip_valid)
validpredicts.show()