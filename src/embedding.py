from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os
from utils import save_pkl
from data_preprocessing import clean_data
from logger import logging
import pandas as pd


def sentence_transformer_embeddings(content):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_vectors = content["tags"].apply(lambda x: model.encode(" ".join(x)))
    sentence_matrix = np.stack(sentence_vectors.values)
    sentence_sim = cosine_similarity(sentence_matrix)
    save_pkl(sentence_sim, "sentence_similarity.pkl")
    return sentence_sim


def word2vec_embeddings(content):
    corpus = content["tags"].apply(lambda x: x if isinstance(x, list) else x.split())
    w2v_model = Word2Vec(corpus, vector_size=50, window=5, min_count=1, workers=4)

    def get_w2v_vector(words):
        vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if len(vectors) == 0:
            return np.zeros(w2v_model.vector_size)
        return np.mean(vectors, axis=0)

    w2v_matrix = np.array([get_w2v_vector(words) for words in corpus])
    w2v_sim = cosine_similarity(w2v_matrix)
    save_pkl(w2v_sim, "w2v_similarity.pkl")
    return w2v_sim


logging.info(" Data embedding completed successfully!")


content = pd.read_csv(
    r"D:\PROJECTS\ML\Recommendation-System\data\processed\processed.csv"
)

print("Running embeddings...")
sentence_transformer_embeddings(content)
word2vec_embeddings(content)
