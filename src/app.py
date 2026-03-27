import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# -----------------------------
# Setup NLTK
# -----------------------------
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# -----------------------------
# Load dataset and Hybrid artifacts
# -----------------------------
@st.cache_resource
def load_models():
    # Use relative paths
    content = pd.read_csv(os.path.join("data", "processed", "processed.csv"))
    
    with open(os.path.join("data", "artifacts", "tfidf_matrix.pkl"), "rb") as f:
        tfidf_matrix = pickle.load(f)
        
    with open(os.path.join("data", "artifacts", "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf_vectorizer = pickle.load(f)
        
    with open(os.path.join("data", "artifacts", "w2v_matrix.pkl"), "rb") as f:
        w2v_matrix = pickle.load(f)
        
    w2v_model = Word2Vec.load(os.path.join("data", "artifacts", "w2v_model.bin"))
    
    return content, tfidf_matrix, tfidf_vectorizer, w2v_matrix, w2v_model


content, tfidf_matrix, tfidf_vectorizer, w2v_matrix, w2v_model = load_models()


# -----------------------------
# Helpers: Vectorization
# -----------------------------
def get_tfidf_query_vec(input_text):
    words = input_text.lower().split()
    clean_text = " ".join([lemmatizer.lemmatize(word) for word in words if word not in stop_words])
    return tfidf_vectorizer.transform([clean_text])

def get_w2v_query_vec(input_text):
    words = input_text.lower().split()
    clean_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    matched_vectors = [w2v_model.wv[word] for word in clean_words if word in w2v_model.wv]
    if not matched_vectors:
        return np.zeros(w2v_model.vector_size)
    return np.mean(matched_vectors, axis=0)


# -----------------------------
# Recommendation function (Hybrid)
# -----------------------------
def recommend_product_images(input_text, top_k=5, tfidf_weight=0.5):
    # Calculate TF-IDF similarities
    tfidf_q = get_tfidf_query_vec(input_text)
    tfidf_sims = cosine_similarity(tfidf_q, tfidf_matrix)[0]
    
    # Calculate Word2Vec similarities
    w2v_q = get_w2v_query_vec(input_text).reshape(1, -1)
    w2v_sims = cosine_similarity(w2v_q, w2v_matrix)[0]
    
    # Combine scores (Hybrid)
    hybrid_sims = (tfidf_weight * tfidf_sims) + ((1 - tfidf_weight) * w2v_sims)
    
    # Get top K
    top_indices = hybrid_sims.argsort()[-top_k:][::-1]

    recommended_products = content["name"].iloc[top_indices].tolist()
    image_urls = content["img"].iloc[top_indices].tolist()
    return recommended_products, image_urls


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hybrid Recommender", layout="wide")

st.title("🛍️ Hybrid Product Recommender")
st.write("Combining **TF-IDF Keyword Accuracy** with **Word2Vec Semantic Intelligence**.")

with st.sidebar:
    st.header("Search Settings")
    weight = st.slider(
        "Similarity Balance", 
        0.0, 1.0, 0.6, 
        help="Higher values favor exact keyword matches (Brand/Name). Lower values favor similar categories (Semantics)."
    )
    st.write(f"🔤 Keywords: {weight*100:.0f}%")
    st.write(f"🧠 Semantics: {(1-weight)*100:.0f}%")
    
    top_k = st.slider("Number of results", 1, 20, 5)

input_text = st.text_input(
    "Search for products...",
    placeholder="e.g., Nike shoes, ethnic kurti, leather bag",
)

if st.button("Search", type="primary"):
    if not input_text.strip():
        st.error("Please enter a search query.")
    else:
        with st.spinner("Finding the best matches..."):
            recommended_products, image_urls = recommend_product_images(
                input_text, top_k=top_k, tfidf_weight=weight
            )
        
        st.subheader(f"Top matches for '{input_text}':")

        cols = st.columns(len(recommended_products))
        for col, name, img_url in zip(cols, recommended_products, image_urls):
            with col:
                st.image(img_url, use_container_width=True, caption=name)


