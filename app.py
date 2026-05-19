import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

# Page Configuration
st.set_page_config(page_title="Anime Recommender", page_icon="📺", layout="wide")

# 1. Caching Model & Data
# Using @st.cache_resource to prevent .pkl files from reloading on every interaction
@st.cache_resource
def load_resources():
    # Ensure the 'models' folder is in the same directory as app.py in GitHub
    tfidf_genre = joblib.load('models/tfidf_genre.pkl')
    tfidf_type = joblib.load('models/tfidf_type.pkl')
    genre_matrix = joblib.load('models/genre_matrix.pkl')
    type_matrix = joblib.load('models/type_matrix.pkl')
    df = joblib.load('models/anime_df.pkl')
    
    return tfidf_genre, tfidf_type, genre_matrix, type_matrix, df

# Load models and handle potential FileNotFoundError
try:
    tfidf_genre, tfidf_type, genre_matrix, type_matrix, df = load_resources()
except FileNotFoundError:
    st.error("Model files not found. Please ensure the 'models' directory is pushed to GitHub.")
    st.stop()

# Extract unique choices for the user interface
# Get all unique genres from the preprocessed dataframe
unique_genres = sorted(list(set([g for sublist in df['genre'].str.split() for g in sublist if isinstance(g, str)])))
# Get all unique types
unique_types = sorted([t.capitalize() for t in df['type'].unique() if isinstance(t, str)])

# 2. Core Recommendation Function
def get_recommendations(genres, anime_type, weight_genre, weight_type, min_rating, min_members, top_n):
    # Build Query
    genre_query = ' '.join(genres).lower()
    type_query = anime_type.lower().strip()

    # Transform into TF-IDF space
    user_genre_vec = tfidf_genre.transform([genre_query])
    user_type_vec = tfidf_type.transform([type_query])

    # Calculate Cosine Similarity
    genre_sim = cosine_similarity(user_genre_vec, genre_matrix).flatten()
    type_sim = cosine_similarity(user_type_vec, type_matrix).flatten()

    # Weighted blend (Dynamic based on user input)
    final_score = (weight_genre * genre_sim) + (weight_type * type_sim)
    
    # Copy DataFrame to avoid modifying the original data
    results_df = df.copy()
    results_df['score'] = final_score

    # Apply Filters for rating and members
    results_df = results_df[results_df['rating'] >= min_rating]
    results_df = results_df[results_df['members'] >= min_members]
    
    # Sort by highest score and return the top N results
    results_df = results_df.sort_values('score', ascending=False)
    
    return results_df.head(top_n)[['name', 'genre', 'type', 'rating', 'members', 'score']]

# 3. Streamlit User Interface (UI)
st.title("📺 Anime Recommendation System")
st.markdown("Temukan anime berdasarkan preferensi genre dan format tayangan Anda menggunakan pendekatan *Content-Based Filtering*.")

st.sidebar.header("⚙️ Konfigurasi Rekomendasi")

# Sidebar: Basic Filters
st.sidebar.subheader("Kriteria Dasar")
selected_genres = st.sidebar.multiselect("Pilih Genre:", unique_genres, default=["romance", "drama"])
selected_type = st.sidebar.selectbox("Pilih Tipe (Format):", unique_types, index=unique_types.index("Movie") if "Movie" in unique_types else 0)

# Sidebar: Advanced Filters (Rating & Members)
st.sidebar.subheader("Filter Kelayakan")
min_rating_input = st.sidebar.slider("Minimal Rating", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
min_members_input = st.sidebar.number_input("Minimal Members", min_value=0, value=5000, step=1000)

# Sidebar: System Weights
st.sidebar.subheader("Pengaturan Bobot (Advanced)")
st.sidebar.markdown("Atur seberapa besar pengaruh Genre vs Tipe terhadap hasil akhir.")
w_genre = st.sidebar.slider("Bobot Genre", 0.0, 1.0, 0.7, 0.1)
w_type = st.sidebar.slider("Bobot Tipe", 0.0, 1.0, 0.3, 0.1)
top_n_input = st.sidebar.slider("Jumlah Rekomendasi", 5, 50, 10)

# Main Output Area
if st.sidebar.button("Cari Rekomendasi 🚀", type="primary"):
    if not selected_genres:
        st.warning("Harap pilih setidaknya satu genre.")
    else:
        st.markdown(f"### Menampilkan Top {top_n_input} Rekomendasi")
        st.write(f"**Kueri Anda:** Genre `{' | '.join(selected_genres)}`, Tipe `{selected_type}`")
        
        with st.spinner('Menghitung tingkat kemiripan...'):
            recommendations = get_recommendations(
                genres=selected_genres,
                anime_type=selected_type,
                weight_genre=w_genre,
                weight_type=w_type,
                min_rating=min_rating_input,
                min_members=min_members_input,
                top_n=top_n_input
            )
            
        if recommendations.empty:
            st.info("Tidak ditemukan anime yang cocok dengan kriteria filter Anda. Coba turunkan Minimal Rating atau Minimal Members.")
        else:
            # Format the dataframe for better readability
            st.dataframe(
                recommendations.style.format({
                    "rating": "{:.2f}",
                    "members": "{:,}",
                    "score": "{:.4f}"
                }),
                use_container_width=True,
                hide_index=True
            )
