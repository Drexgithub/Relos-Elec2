from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, round, spark_partition_id, min, max

# 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("StudentPerformance_BigData_Analysis") \
    .getOrCreate()

# 2. Load the dataset
df = spark.read.csv("StudentPerformanceFactors.csv", header=True, inferSchema=True)
print(f"Total records loaded: {df.count()}") 

# 3. Data Cleaning
df_clean = df.na.drop()

# 4. Strategy: Hash Partitioning (Logical)
df_partitioned = df_clean.repartition(4, "Gender")

# 5. Analysis: Group by Parental Education Level
education_impact = df_partitioned.groupBy("Parental_Education_Level") \
    .agg(round(avg("Exam_Score"), 2).alias("Avg_Score")) \
    .orderBy(col("Avg_Score").desc())

print("\n--- Impact of Parental Education Level ---")
education_impact.show()

# 6. Strategy: Directory Partitioning (Physical/On-Disk)
df_clean.write.partitionBy("School_Type").mode("overwrite").csv("student_output_data")
print("Directory Partitioning Complete: Folders created in 'student_output_data'")

# 7. Strategy: Range Partitioning
df_range = df_clean.repartitionByRange(3, "Exam_Score")
df_range.createOrReplaceTempView("student_table")

# 8. SQL Range Analysis
print("\n--- Analysis of Partition Score Ranges ---")
spark.sql("""
    SELECT
        spark_partition_id() as partition_id,
        min(Exam_Score) as min_score_in_part,
        max(Exam_Score) as max_score_in_part,
        count(*) as student_count
    FROM student_table
    GROUP BY partition_id
    ORDER BY partition_id
""").show()

spark.stop()
