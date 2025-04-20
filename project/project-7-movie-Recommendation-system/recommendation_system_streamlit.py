import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Set page title
st.title("🎬 Movie Recommendation System")

# Load MovieLens 100K dataset from extracted files in the same directory
@st.cache_data
def load_data():
    # Load u.data (userId, movieId, rating, timestamp) - tab-separated, no header
    ratings = pd.read_csv('u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
    # Load u.item (movieId, title, etc.) - | separated, take first two columns
    movies = pd.read_csv('u.item', sep='|', encoding='latin-1', usecols=[0, 1], names=['movieId', 'title'])
    return ratings, movies

ratings, movies = load_data()

# Create user-movie matrix
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Calculate cosine similarity between users
@st.cache_data
def compute_similarity():
    user_similarity = cosine_similarity(user_movie_matrix)
    return pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

user_similarity_df = compute_similarity()

# Function to recommend movies
def recommend_movies(user_id, n=5):
    if user_id not in user_movie_matrix.index:
        return pd.DataFrame(columns=['movieId', 'title', 'predicted_rating'])
    
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:n+1].index
    similar_users_ratings = user_movie_matrix.loc[similar_users]
    avg_ratings = similar_users_ratings.mean()
    user_rated_movies = user_movie_matrix.loc[user_id]
    unseen_movies = avg_ratings[user_rated_movies == 0]
    top_movies = unseen_movies.sort_values(ascending=False).head(n)
    
    recommendations = movies[movies['movieId'].isin(top_movies.index)][['movieId', 'title']].copy()
    recommendations['predicted_rating'] = top_movies.values
    return recommendations.sort_values(by='predicted_rating', ascending=False)

# Streamlit interface
st.subheader("Get Movie Recommendations")
user_id = st.selectbox("Select User ID", options=sorted(user_movie_matrix.index))
num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

if st.button("Recommend Movies"):
    recommendations = recommend_movies(user_id, num_recommendations)
    if not recommendations.empty:
        st.write(f"Top {num_recommendations} movie recommendations for User {user_id}:")
        st.dataframe(recommendations[['title', 'predicted_rating']].style.format({'predicted_rating': '{:.2f}'}))
    else:
        st.error(f"No recommendations available for User {user_id}.")