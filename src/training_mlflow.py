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


def word2vec_embeddings(content, vector_size=50, window=5, min_count=1):
    corpus = content["tags"].apply(lambda x: x if isinstance(x, list) else x.split())
    w2v_model = Word2Vec(
        corpus, vector_size=vector_size, window=window, min_count=min_count, workers=4
    )

    def get_w2v_vector(words):
        vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if len(vectors) == 0:
            return np.zeros(w2v_model.vector_size)
        return np.mean(vectors, axis=0)

    w2v_matrix = np.array([get_w2v_vector(words) for words in corpus])
    w2v_sim = cosine_similarity(w2v_matrix)
    save_pkl(w2v_sim, "w2v_similarity.pkl")  # stored in project root
    return w2v_sim, vector_size, window, min_count


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")  # local tracking
    mlflow.set_experiment("Content-Based Recommender")

    content = pd.read_csv(os.path.join("data", "processed", "processed.csv"))
    num_items = len(content)

    with mlflow.start_run():
        start_time = time.time()

        # --- Word2Vec similarity generation ---
        w2v_sim, vector_size, window, min_count = word2vec_embeddings(content)

        # Log params
        mlflow.log_param("vector_size", vector_size)
        mlflow.log_param("window", window)
        mlflow.log_param("min_count", min_count)

        # Log metrics
        mlflow.log_metric("w2v_avg_similarity", float(w2v_sim.mean()))
        mlflow.log_metric("num_items", num_items)

        # Log artifact (model matrix)
        mlflow.log_artifact("data/artifacts/w2v_similarity.pkl")

        # Save dataset also
        mlflow.log_artifact(os.path.join("data", "processed", "processed.csv"))

        total_time = round(time.time() - start_time, 3)
        mlflow.log_metric("execution_time_sec", total_time)

        logging.info("✅ Word2Vec completed and logged to MLflow")
        print(f"✅ MLflow run completed in {total_time} sec")
        print("➡️ View results: mlflow ui")
