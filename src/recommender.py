from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def train_als_model(training_data):
    print("Initializing Distributed ALS Recommendation Model...")
    
    # implicitPrefs=True is crucial! It tells the model we are using purchase counts, not 1-5 star ratings.
    als = ALS(maxIter=10, 
              regParam=0.01, 
              userCol="user_id", 
              itemCol="product_id", 
              ratingCol="purchase_count",
              implicitPrefs=True,
              coldStartStrategy="drop")
              
    print("Training model across Spark cluster...")
    model = als.fit(training_data)
    
    return model

def generate_recommendations(model, num_items=5):
    print(f"Generating top {num_items} product recommendations for all users...")
    user_recs = model.recommendForAllUsers(num_items)
    return user_recs
