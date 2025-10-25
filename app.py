import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.data_preprocessing import clean_data

# Load data and embeddings
# content = clean_data()  # DataFrame with columns: 'name', 'img', 'tags'

with open("artifacts/w2v_similarity.pkl", "rb") as f:
    w2v_matrix = pickle.load(f)

with open("artifacts/sentence_similarity.pkl", "rb") as f:
    st_matrix = pickle.load(f)


# Helper: approximate vector for arbitrary input
def approximate_input_vector(input_text, method="w2v"):
    content = clean_data()
    words = input_text.lower().split()
    if method == "w2v":
        matched_vectors = [
            w2v_matrix[i]
            for i, tags in enumerate(content["tags"])
            if any(word in str(tags).lower().split() for word in words)
        ]
        return (
            np.mean(matched_vectors, axis=0)
            if matched_vectors
            else np.mean(w2v_matrix, axis=0)
        )
    else:
        matched_vectors = [
            st_matrix[i]
            for i, tags in enumerate(content["tags"])
            if any(word in str(tags).lower().split() for word in words)
        ]
        return (
            np.mean(matched_vectors, axis=0)
            if matched_vectors
            else np.mean(st_matrix, axis=0)
        )


# Recommendation function
def recommend_product_images(input_text, method="w2v", top_k=5):
    input_vec = approximate_input_vector(input_text, method=method).reshape(1, -1)
    sims = cosine_similarity(input_vec, w2v_matrix if method == "w2v" else st_matrix)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return content.iloc[top_indices][["name", "img"]]


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Product Recommendation System")
query = st.text_input("Enter product name or description:")

if query:
    st.subheader("Word2Vec Recommendations")
    w2v_recs = recommend_product_images(query, method="w2v")
    for _, row in w2v_recs.iterrows():
        st.image(row["img"], caption=row["name"], width=200)

    st.subheader("Sentence Transformer Recommendations")
    st_recs = recommend_product_images(query, method="st")
    for _, row in st_recs.iterrows():
        st.image(row["img"], caption=row["name"], width=200)
