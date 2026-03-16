from pyspark.sql import SparkSession
from pyspark.sql.functions import count

def prepare_implicit_data(spark, data_path):
    print("Loading grocery order data...")
    # Expecting a CSV with 'user_id' and 'product_id'
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    print("Aggregating purchase history for implicit feedback...")
    # Group by user and product to get the total number of times they bought it
    # This 'purchase_count' becomes our implicit "rating"
    user_product_df = df.groupBy("user_id", "product_id") \
                        .agg(count("*").alias("purchase_count"))
    
    # Filter out one-time purchases to reduce noise and compute time
    filtered_df = user_product_df.filter(user_product_df.purchase_count > 1)
    
    return filtered_df
