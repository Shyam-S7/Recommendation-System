import mlflow
import mlflow.sklearn
import time
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from utils import save_pkl
from logger import logging
import os
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import ast

def hybrid_vectorization(content):
    # Correctly parse the list-like strings from CSV back into actual lists
    content["tags"] = content.get("tags", pd.Series([])).apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    corpus = content["tags"]
    
    # --- 1. TF-IDF for keyword matching ---
    corpus_strings = corpus.apply(lambda x: " ".join(x))
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_strings)
    
    # --- 2. Word2Vec for semantic similarity ---
    w2v_model = Word2Vec(
        corpus, vector_size=100, window=5, min_count=1, workers=4, 
        sg=1, epochs=30
    )
    def get_w2v_vector(words):
        vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)
    w2v_matrix = np.array([get_w2v_vector(words) for words in corpus])
    
    # Save all artifacts
    save_pkl(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    save_pkl(tfidf_matrix, "tfidf_matrix.pkl")
    save_pkl(w2v_matrix, "w2v_matrix.pkl")
    w2v_model.save("data/artifacts/w2v_model.bin")
    
    return tfidf_matrix, tfidf_vectorizer, w2v_matrix, w2v_model


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")  # local tracking
    mlflow.set_experiment("Content-Based Recommender")

    content = pd.read_csv(os.path.join("data", "processed", "processed.csv"))
    num_items = len(content)

    with mlflow.start_run():
        start_time = time.time()

        # --- Hybrid generation ---
        tfidf_matrix, tfidf_vec, w2v_matrix, w2v_model = hybrid_vectorization(content)

        # Log params
        mlflow.log_param("model_type", "Hybrid (TF-IDF + Word2Vec)")
        mlflow.log_param("w2v_epochs", 30)
        mlflow.log_param("tfidf_features", 5000)

        # Log metrics
        mlflow.log_metric("num_items", num_items)
        mlflow.log_metric("vocabulary_size", len(tfidf_vec.vocabulary_))

        # Log all artifacts
        mlflow.log_artifact("data/artifacts/tfidf_vectorizer.pkl")
        mlflow.log_artifact("data/artifacts/tfidf_matrix.pkl")
        mlflow.log_artifact("data/artifacts/w2v_matrix.pkl")
        mlflow.log_artifact("data/artifacts/w2v_model.bin")

        # Save dataset also
        mlflow.log_artifact(os.path.join("data", "processed", "processed.csv"))

        total_time = round(time.time() - start_time, 3)
        mlflow.log_metric("execution_time_sec", total_time)

        logging.info("✅ Hybrid vectorization completed")
        print(f"✅ MLflow run completed in {total_time} sec")
        print("➡️ View results: mlflow ui")



