import os
from pyspark.sql import SparkSession
from data_prep import prepare_implicit_data
from recommender import train_als_model, generate_recommendations

def run_ecommerce_pipeline():
    # Provide path to a sample dataset for the portfolio
    data_path = "../data/order_products.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Please place order data at {data_path}")
        return

    spark = SparkSession.builder \
        .appName("Lidl_Commerce_Recommender") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    try:
        # 1. Prep the Data
        interactions_df = prepare_implicit_data(spark, data_path)
        
        # Split data for training
        (training, test) = interactions_df.randomSplit([0.8, 0.2])
        
        # 2. Train the Model
        model = train_als_model(training)
        
        # 3. Generate Recommendations
        recommendations = generate_recommendations(model, num_items=5)
        
        print("\n================ SAMPLE RECOMMENDATIONS ================")
        # Show recommendations for 5 random users
        recommendations.show(5, truncate=False)
        print("Pipeline execution complete.")

    except Exception as e:
        print(f"Pipeline failed: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    run_ecommerce_pipeline()
