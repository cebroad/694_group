from pyspark import SparkContext, SparkConf
from pyspark.sql import Row
from datetime import datetime
from pyspark.sql.types import *
from pyspark.sql.window import Window
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
        , toIntSafe(r[1])
        , toDateTimeSafe(r[2])
        , r[3]
        , toIntSafe(r[4])
        , toDateTimeSafe(r[5])
        , r[6]
        , toIntSafe(r[7])
        , toIntSafe(r[8])
        , r[9]
        , r[10]
    )


def importAndConvert(file_name, conv_function, schema_name, ):
    # ### Import YYYYMM_station_data.csv and convert to Data Frame

    # Step 1: Import and remove header for August Data
    data = sc.textFile(file_name)
    header = data.first()  # extract header
    data = data.filter(lambda row: row != header)  # filter to exclude header
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
    # variable newdf will be updated several times
    newdf = df
    for c in cols:
        si = StringIndexer(inputCol=c, outputCol=c + "-num")
        sm = si.fit(newdf)
        newdf = sm.transform(newdf).drop(c)
        newdf = newdf.withColumnRenamed(c + "-num", c)
    return newdf


conf = SparkConf().setMaster("local").setAppName("Bike_Share")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# Defining Schemas

tripSchema = StructType([
    StructField("trip_id", IntegerType(), True),
    StructField("duration", IntegerType(), True),
    StructField("start_date", TimestampType(), True),
    StructField("start_station", StringType(), True),
    StructField("start_terminal", IntegerType(), True),
    StructField("end_date", TimestampType(), True),
    StructField("end_station", StringType(), True),
    StructField("end_terminal", IntegerType(), True),
    StructField("bike_no", IntegerType(), True),
    StructField("sub_type", StringType(), True),
    StructField("zip", StringType(), True)
])

# Trip Data
trip_201408_DF = importAndConvert('data/201408_trip_data.csv', convTripString, tripSchema)
trip_201402_DF = importAndConvert('data/201402_trip_data.csv', convTripString, tripSchema)

trip_12mos = trip_201408_DF.unionAll(trip_201402_DF)

print "Dataset 1 Count: ", trip_201408_DF.count()
print "Dataset 2 Count: ", trip_201402_DF.count()
print "Full 12 MOnths Dataset Count: ", trip_12mos.count()

# trip_201402_DF.show()
trip_12mos.show()

# Feature: Average ride duration for station

trip_12mos_step1 = trip_12mos.withColumn('average', avg(trip_12mos.duration).over(
    Window.partitionBy(trip_12mos.start_station)).alias('average_duration'))


# Feature: Round trip

def sameVal(x, y):
    if x == y:
        return 1
    else:
        return 0


round_trip = udf(lambda x, y: sameVal(x, y))

trip_12mos_step2 = trip_12mos_step1.withColumn('round_trip', round_trip('start_station', 'end_station'))

trip_12mos_step2.show()


# Feature: day of week
# Monday is 0 and Sunday is 6

def calc_weekday_weekend(x):
    if x in set([0, 1, 2, 3, 4]):
        return 'weekday'
    else:
        return 'weekend'


def calc_dow(dataFrame):
    day_of_week = udf(lambda (x): x.weekday())
    week_day_end = udf(lambda (x): calc_weekday_weekend(x))
    dataFrame = dataFrame.withColumn('start_day', day_of_week('start_date'))
    dataFrame = dataFrame.withColumn('start_dow', week_day_end('start_day'))
    dataFrame.show()
    dataFrame.groupBy('start_day').count().orderBy('start_day').show()
    return dataFrame

trip_12mos_step3 = calc_dow(trip_12mos_step2)

# Feature: Pct of weekday/weekend trips by station

trip_12mos_step3.write.saveAsTable("trip_12mos_step3")

station_df = sqlContext.sql("""select start_station, sum(case when start_dow = 'weekday' then 1.0 else 0.0 end) as weekday_trips, sum(case when start_dow = 'weekend' then 1.0 else 0.0 end) as weekend_trips ,count(*) as total_trips from trip_12mos_step3 group by start_station""")
station_df2 = station_df.select(station_df['start_station']
                    , round(station_df['weekday_trips']/station_df['total_trips'],2).alias('pct_weekday')
                    , round(station_df['weekend_trips']/station_df['total_trips'],2).alias('pct_weekend'))
trip_12mos_step4 = trip_12mos_step3.join(station_df2, station_df2.start_station == trip_12mos_step3.start_station, 'left_outer').drop(station_df2.start_station)

# END FEATURES

trip_12mos_full = trip_12mos_step4

# trip_12mos_full.take(5)

# Convert strings to numeric values for evaluation (i.e. "Subscriber" = 1, "Customer" = 0)
trip_numeric = indexStringColumns(trip_12mos_full, ["sub_type", "start_station", "end_station",
                                                    "zip"])
# trip_numeric = trip_numeric.drop("start_date", "end_date")  # Can add these back in after one-hot encoding them
# trip_numeric.show()


# Create feature vector
input_cols=["duration","round_trip","average","start_day","start_dow"] #add other features here
va = VectorAssembler(outputCol="features", inputCols=input_cols)
trip_points = va.transform(trip_numeric).select("features", "sub_type").withColumnRenamed("sub_type", "label")
# Create Training and Test data.
triptsets = trip_points.randomSplit([0.8, 0.2])
trip_train = triptsets[0].cache()
trip_valid = triptsets[1].cache()
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(trip_train)
validpredicts = lrModel.transform(trip_valid)
validpredicts.show()
log_rdd = validpredicts.select("prediction", "label").rdd
log_mm = MulticlassMetrics(log_rdd)
log_mm.precision()
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(validpredicts)
print accuracy