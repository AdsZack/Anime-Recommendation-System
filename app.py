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
# Removed weight_genre and weight_type from parameters
def get_recommendations(genres, anime_type, min_rating, min_members, top_n):
    # Build Query
    genre_query = ' '.join(genres).lower()
    type_query = anime_type.lower().strip()

    # Transform into TF-IDF space
    user_genre_vec = tfidf_genre.transform([genre_query])
    user_type_vec = tfidf_type.transform([type_query])

    # Calculate Cosine Similarity
    genre_sim = cosine_similarity(user_genre_vec, genre_matrix).flatten()
    type_sim = cosine_similarity(user_type_vec, type_matrix).flatten()

    # Weighted blend (Fixed default weights: Genre 0.7, Type 0.3)
    final_score = (0.7 * genre_sim) + (0.3 * type_sim)
    
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
st.markdown("Find Your Anime!")

st.sidebar.header("⚙️ Recommendation Configuration")

# Sidebar: Basic Filters
st.sidebar.subheader("Basic Criteria")
selected_genres = st.sidebar.multiselect("Choose Genre:", unique_genres)
selected_type = st.sidebar.selectbox("Choose Type Anime:", unique_types)

# Sidebar: Advanced Filters (Rating & Members)
st.sidebar.subheader("Filtering")
min_rating_input = st.sidebar.slider("Minimal Rating", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
min_members_input = st.sidebar.number_input("Minimal Members", min_value=0, value=5000, step=1000)

# Sidebar: Output Settings
st.sidebar.subheader("Output Settings")
top_n_input = st.sidebar.slider("Total Recommendation", 5, 50, 10)

# Main Output Area
if st.sidebar.button("Find Recommendation 🚀", type="primary"):
    if not selected_genres:
        st.warning("Please choose at least one Genre.")
    else:
        st.markdown(f"### Show Top {top_n_input} Recommendations")
        st.write(f"**Your Query:** Genre `{' | '.join(selected_genres)}`, Type `{selected_type}`")
        
        with st.spinner('Calculating Similarity...'):
            # Removed weight arguments from the function call
            recommendations = get_recommendations(
                genres=selected_genres,
                anime_type=selected_type,
                min_rating=min_rating_input,
                min_members=min_members_input,
                top_n=top_n_input
            )
            
        if recommendations.empty:
            st.info("No anime were found that match your filter criteria. Try lowering the Minimum Rating or Minimum Members.")
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
