from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, sum, desc, unix_timestamp
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from geopy.distance import geodesic
from pyspark.sql.window import Window
from pyspark.sql.functions import lag
 
# Sukuriame Spark sesiją
spark = SparkSession.builder \
  .appName("ShipDistanceAnalysis") \
  .getOrCreate()
 
# Apibrėžiame duomenų schemą, imame tik pirmus 5 stulpelius, nes kiti duomenys mums nereikalingi skaičiavimams
schema = StructType([
  StructField("Timestamp", StringType(), True),
  StructField("Type", StringType(), True),
  StructField("MMSI", StringType(), True),
  StructField("Latitude", FloatType(), True),
  StructField("Longitude", FloatType(), True)
])
 
# Įkeliame duomenis
data = spark.read.csv("data/aisdk-2024-05-04.csv", schema=schema, header=True)

# Konvertuojame Timestamp į unix time (sekundėmis)
data = data.withColumn("Timestamp", unix_timestamp(col("Timestamp"), "dd/MM/yyyy HH:mm:ss"))

# Nufiltruojame neteisingas platumos ir ilgumos reikšmes
data = data.filter((col("Latitude") >= -90.0) & (col("Latitude") <= 90.0))
data = data.filter((col("Longitude") >= -180.0) & (col("Longitude") <= 180.0))

# Apibrėžiame UDF funkciją atstumui apskaičiuoti naudojant geodesic formulę
# Kai kurie laivai turi blogus duomenys, pvz. platuma yra tai 9.164768, tai 89.16481;
# tokie laivai tariamai nuplaukia didžiulį atstumą, kurio nuplaukti neįmanoma per dieną (pvz. MMSI 219000962 "nuplaukia" 91785 km)
# todėl reikia patikrinti, ar laivo greitis yra realistiškas
def geodesic_distance(ts1, ts2, lat1, lon1, lat2, lon2):
  # lat ir lon neturi būti tušti
  if None in (lat1, lon1, lat2, lon2):
    return 0.0
  # laikas turi būti išrikiuotas didėjimo tvarka
  if ts1 > ts2:
    raise Exception("Timestamps are not in order")
  kms = geodesic((lat1, lon1), (lat2, lon2)).kilometers # skaičiuojame nuplauktą atstuma kilometrais
  d_t = ts2 - ts1 # skaičiuojame laiko skirtumą sekundėmis
  # jei laikas yra vienodas, laikome, kad laivas nepajudėjo
  if d_t == 0:
    return 0.0
  speed = (kms / d_t) * 3600 # skaičiuojame greitį km/val
  if speed > 107.0: # laivai negali plaukti greičiau, nei 107 km/h (https://en.wikipedia.org/wiki/HSC_Francisco)
    return 0.0
  return kms

geodesic_udf = udf(geodesic_distance, FloatType())

# Sukuriame lango specifikacijas atstumui apskaičiuoti tarp nuoseklių pozicijų
windowSpec = Window.partitionBy("MMSI").orderBy("Timestamp")

data = data.withColumn("PrevLatitude", lag("Latitude").over(windowSpec)) \
           .withColumn("PrevLongitude", lag("Longitude").over(windowSpec)) \
           .withColumn("PrevTimestamp", lag("Timestamp").over(windowSpec))

# Pritaikome UDF funkciją atstumui apskaičiuoti
data = data.withColumn("Distance", geodesic_udf(col("PrevTimestamp"), col("Timestamp"), col("Latitude"), col("Longitude"), col("PrevLatitude"), col("PrevLongitude")))

# Agreguojame atstumus pagal MMSI ir ieškome didžiausio nuplaukto atstumo
max_distance_ship = data.groupBy("MMSI").agg(sum("Distance").alias("TotalDistance")).orderBy(desc("TotalDistance")).first()

# Spausdiname rezultatą
if max_distance_ship:
  print(f"Laivas su MMSI {max_distance_ship['MMSI']} nuplaukė ilgiausią maršrutą: {max_distance_ship['TotalDistance']} km")
else:
  print("No results after distance calculation. Check the data and computations.")

# Sustabdome Spark sesiją
spark.stop()
