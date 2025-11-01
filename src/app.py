import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load dataset and embeddings
# -----------------------------
@st.cache_data
def load_data():
    content = pd.read_csv(
        r"D:\PROJECTS\ML\Recommendation-System\data\processed\processed.csv"
    )
    with open("data/artifacts/w2v_similarity.pkl", "rb") as f:
        w2v_matrix = pickle.load(f)
    with open("data/artifacts/sentence_similarity.pkl", "rb") as f:
        st_matrix = pickle.load(f)
    return content, w2v_matrix, st_matrix


content, w2v_matrix, st_matrix = load_data()


# -----------------------------
# Helper: approximate vector for arbitrary input
# -----------------------------
def approximate_input_vector(input_text, method="w2v"):
    words = input_text.lower().split()

    if method == "w2v":
        matched_vectors = []
        for i, tags in enumerate(content["tags"]):
            if not isinstance(tags, list):
                tags = str(tags).lower().split()
            if any(word in tags for word in words):
                matched_vectors.append(w2v_matrix[i])
        return (
            np.mean(matched_vectors, axis=0)
            if matched_vectors
            else np.mean(w2v_matrix, axis=0)
        )

    elif method == "st":
        matched_vectors = []
        for i, tags in enumerate(content["tags"]):
            if not isinstance(tags, list):
                tags = str(tags).lower().split()
            if any(word in tags for word in words):
                matched_vectors.append(st_matrix[i])
        return (
            np.mean(matched_vectors, axis=0)
            if matched_vectors
            else np.mean(st_matrix, axis=0)
        )
    else:
        raise ValueError("Method must be 'w2v' or 'st'")


# -----------------------------
# Recommendation function
# -----------------------------
def recommend_product_images(input_text, method="w2v", top_k=5):
    input_vec = approximate_input_vector(input_text, method=method).reshape(1, -1)
    sims = cosine_similarity(input_vec, w2v_matrix if method == "w2v" else st_matrix)[0]
    top_indices = sims.argsort()[-top_k:][::-1]

    recommended_products = content["name"].iloc[top_indices].tolist()
    image_urls = content["img"].iloc[top_indices].tolist()
    return recommended_products, image_urls


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🛍️ Product Recommender System")
st.write("Enter a product name or description to get similar product recommendations.")

input_text = st.text_input(
    "Product Description",
    placeholder="e.g., red shoes, leather bag, wireless headphones",
)

method = st.selectbox("Select Similarity Method", ["w2v", "st"])
top_k = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    if not input_text.strip():
        st.warning("Please enter a product description.")
    else:
        recommended_products, image_urls = recommend_product_images(
            input_text, method=method, top_k=top_k
        )
        st.subheader(f"Recommendations for '{input_text}' using **{method.upper()}**:")

        # Display results in columns
        cols = st.columns(len(recommended_products))
        for col, name, img_url in zip(cols, recommended_products, image_urls):
            with col:
                st.image(img_url, use_container_width=True, caption=name)
