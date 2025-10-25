import pandas as pd
import numpy as np
import pickle
from logger import logging
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, HTML
from data_preprocessing import clean_data  # your cleaning function


# -----------------------------
# Display images with names
# -----------------------------
def show_images_with_names(products, urls):
    html = ""
    for name, url in zip(products, urls):
        html += f"""
        <div style='display:inline-block; text-align:center; margin:10px;'>
            <img src='{url}' width='200' height='200' style='border-radius:10px;'><br>
            <span style='font-size:14px;'>{name}</span>
        </div>
        """
    display(HTML(html))


# -----------------------------
# Load dataset and embeddings
# -----------------------------
content = clean_data()  # DataFrame with columns: 'name', 'img', 'tags'

with open("artifacts/w2v_similarity.pkl", "rb") as f:
    w2v_matrix = pickle.load(f)

with open("artifacts/sentence_similarity.pkl", "rb") as f:
    st_matrix = pickle.load(f)


# -----------------------------
# Helper: approximate vector for arbitrary input
# -----------------------------
def approximate_input_vector(input_text, method="w2v"):
    words = input_text.lower().split()

    if method == "w2v":
        # Average vectors of products whose tags match any word
        matched_vectors = []
        for i, tags in enumerate(content["tags"]):
            if not isinstance(tags, list):
                tags = str(tags).lower().split()
            if any(word in tags for word in words):
                matched_vectors.append(w2v_matrix[i])
        if matched_vectors:
            return np.mean(matched_vectors, axis=0)
        else:
            # fallback: average of all product vectors
            return np.mean(w2v_matrix, axis=0)

    elif method == "st":
        # Same logic for sentence transformer embeddings
        matched_vectors = []
        for i, tags in enumerate(content["tags"]):
            if not isinstance(tags, list):
                tags = str(tags).lower().split()
            if any(word in tags for word in words):
                matched_vectors.append(st_matrix[i])
        if matched_vectors:
            return np.mean(matched_vectors, axis=0)
        else:
            return np.mean(st_matrix, axis=0)
    else:
        raise ValueError("Method must be 'w2v' or 'st'")


# -----------------------------
# Recommendation function
# -----------------------------
def recommend_product_images(input_text, method="w2v", top_k=5):
    # Compute approximate vector for input
    input_vec = approximate_input_vector(input_text, method=method).reshape(1, -1)

    # Compute similarity with all products
    sims = cosine_similarity(input_vec, w2v_matrix if method == "w2v" else st_matrix)[0]

    # Get top-k similar products
    top_indices = sims.argsort()[-top_k:][::-1]
    recommended_products = content["name"].iloc[top_indices].tolist()
    image_urls = content["img"].iloc[top_indices].tolist()

    print(f"Recommendations for '{input_text}' using {method.upper()}:")
    show_images_with_names(recommended_products, image_urls)


# -----------------------------
# Example usage
# -----------------------------
product_description = input("Enter any product description or name: ")
recommend_product_images(product_description, method="w2v", top_k=5)
recommend_product_images(product_description, method="st", top_k=5)



if __name__='__main___':