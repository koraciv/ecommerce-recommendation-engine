# E-Commerce Recommendation Engine (PySpark)

## 📌 Project Overview
In online grocery and retail commerce, personalized recommendations are the primary driver for increasing Average Order Value (AOV) and improving customer retention. 

This project implements a scalable **Collaborative Filtering Recommendation Engine** using **Apache Spark (PySpark MLlib)**. Using an online grocery dataset (Instacart), it processes millions of transactional records to extract implicit user preferences (purchase frequencies) and predict the top products a customer is most likely to buy next.

**Business Value:** Simulates the core algorithmic logic required for "Frequently Bought Together" and "Recommended for You" carousels in large-scale e-commerce platforms like Lidl or Kaufland.

## 🛠️ Tech Stack
* **Big Data Framework:** Apache Spark, PySpark SQL
* **Machine Learning:** PySpark MLlib (Alternating Least Squares - ALS)
* **Language:** Python 3.x

## 🏗️ Architecture & Workflow
1. **Implicit Data Engineering:** Ingests raw transactional order data. Aggregates user-item interactions to convert raw purchases into implicit "ratings" (purchase frequency).
2. **Distributed Training:** Utilizes PySpark MLlib's `ALS` algorithm configured for `implicitPrefs=True` to handle non-explicit feedback across a distributed cluster.
3. **Recommendation Generation:** Broadcasts the trained matrix factorization model to generate the Top-N product recommendations for every user in the database.

## 📂 Project Structure
```text
├── data/                   # Data directory (Dataset excluded via .gitignore)
├── src/                    
│   ├── data_prep.py        # PySpark ingestion and aggregation logic
│   ├── recommender.py      # MLlib ALS model training and prediction
│   └── main.py             # Orchestration script
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and directories
└── README.md               # Project documentation
