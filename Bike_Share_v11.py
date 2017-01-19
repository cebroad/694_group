
# -------------------------------------- #
# MSAN Intercession Group Project 
# Brigit, Claire, Melanie			
# -------------------------------------- #

# -------------------------------------- #
# CODE PART 1: IMPORT, CLEAN, STACK DATA
# -------------------------------------- #

# -------------------------------------- #
# LIBRARIES
# -------------------------------------- #

from pyspark import SparkContext, SparkConf
from pyspark.sql import Row
from datetime import datetime
from pyspark.sql.types import *
from pyspark.sql import SQLContext

# --------------------------------------------------------#
# ESTABLISH SPARK CONNECTION
# --------------------------------------------------------#
conf = SparkConf().setMaster("local").setAppName("Bike_Share")
sc   = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

# -------------------------------------- #
# DATA CONVERSION FUNCTIONS
# -------------------------------------- #

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

# -------------------------------------- #
# CONVERSION FUNCTIONS
# ONE FOR EACH INPUT DATA SET
# -------------------------------------- #
def convStationString(r):
    return Row(
    toIntSafe(r[0])
    ,r[1]
    ,toFloatSafe(r[2])
    ,toFloatSafe(r[3])
    ,toIntSafe(r[4])
    ,r[5]
    ,toDateSafe(r[6])
    )

def convWeatherString(r):
    return Row(
    toDateSafe(r[0])
    ,toIntSafe(r[1])
    ,toIntSafe(r[2])
    ,toIntSafe(r[3])
    ,toIntSafe(r[4])
    ,toIntSafe(r[5])
    ,toIntSafe(r[6])
    ,toIntSafe(r[7])
    ,toIntSafe(r[8])
    ,toIntSafe(r[9])
    ,toFloatSafe(r[10])
    ,toFloatSafe(r[11])
    ,toFloatSafe(r[12])
    ,toIntSafe(r[13])
    ,toIntSafe(r[14])
    ,toIntSafe(r[15])
    ,toIntSafe(r[16])
    ,toIntSafe(r[17])
    ,toIntSafe(r[18])
    ,toFloatSafe(r[19])
    ,toIntSafe(r[20])
    ,r[21]
    ,toIntSafe(r[22])
    ,r[23]
    )
    
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

# -------------------------------------- #
# SCHEMA DEFINITIONS
# ONE FOR EACH INPUT DATA SET
# -------------------------------------- #

stationSchema = StructType([
  StructField("station_id", IntegerType(), True),
  StructField("station_name", StringType(), True),
  StructField("lat", FloatType(), True),
  StructField("long", FloatType(), True),
  StructField("dock_ct", IntegerType(), True),
  StructField("city", StringType(), True),
  StructField("install_dt", DateType(), True),
  ])

weatherSchema = StructType([
  StructField("Date", DateType(), True),
  StructField("max_temp", IntegerType(), True),
  StructField("mean_temp", IntegerType(), True),
  StructField("min_temp", IntegerType(), True),
  StructField("max_dp", IntegerType(), True),
  StructField("mean_dp", IntegerType(), True),
  StructField("min_dp", IntegerType(), True),
  StructField("max_humidity", IntegerType(), True),
  StructField("mean_humidity", IntegerType(), True),
  StructField("min_humidity", IntegerType(), True),
  StructField("max_pressure", FloatType(), True),
  StructField("mean_pressure", FloatType(), True),
  StructField("min_pressure", FloatType(), True),
  StructField("max_visibility", IntegerType(), True),
  StructField("mean_visibility", IntegerType(), True),
  StructField("min_visibility", IntegerType(), True),
  StructField("max_ws", IntegerType(), True),
  StructField("mean_ws", IntegerType(), True),
  StructField("max_gust", IntegerType(), True),
  StructField("precipitation", FloatType(), True),
  StructField("cloud_cover", IntegerType(), True),
  StructField("events", StringType(), True),
  StructField("wind_dir", IntegerType(), True),
  StructField("zip_code", StringType(), True)
  ])

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
  
# ---------------------------------------------------- #
# IMPORT FUNCTIONS
# ONE FOR EACH INPUT DATA SET
# Note: Each dataset contains data for one full year
# (12 months) but is partitioned into two six-month 
# csv files. Therefore each file needs to be 
# appended to it's buddy to make 1 full year of data
# ---------------------------------------------------- #

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

# ---------------------------------------------------- #
# CALL FUNCTIONS ON DATASETS
# ---------------------------------------------------- #

# station data
station_201408_DF = importAndConvert('data/201408_station_data.csv', convStationString, stationSchema)
station_201402_DF = importAndConvert('data/201402_station_data.csv', convStationString, stationSchema)

station_12mos = station_201408_DF.unionAll(station_201402_DF)

print "Dataset 1 Count: ", station_201408_DF.count()
print "Dataset 2 Count: ", station_201402_DF.count()
print "Full 12 MOnths Dataset Count: ", station_12mos.count()

print station_12mos.show()


# Trip Data

trip_201408_DF = importAndConvert('data/201408_trip_data.csv', convTripString, tripSchema)
trip_201402_DF = importAndConvert('data/201402_trip_data.csv', convTripString, tripSchema)

trip_12mos = trip_201408_DF.unionAll(trip_201402_DF)

print "Dataset 1 Count: ", trip_201408_DF.count()
print "Dataset 2 Count: ", trip_201402_DF.count()
print "Full 12 MOnths Dataset Count: ", trip_12mos.count()

print trip_12mos.show()

# Weather Data

weather_201408_DF = importAndConvert('data/201408_weather_data.csv', convWeatherString, weatherSchema)
weather_201402_DF = importAndConvert('data/201402_weather_data.csv', convWeatherString, weatherSchema)

weather_12mos = weather_201408_DF.unionAll(weather_201402_DF)

print "Dataset 1 Count: ", weather_201408_DF.count()
print "Dataset 2 Count: ", weather_201402_DF.count()
print "Full 12 MOnths Dataset Count: ", weather_12mos.count()

print weather_12mos.show()