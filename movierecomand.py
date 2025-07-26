import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
bollywood = pd.read_csv("bollywood_movies.csv")
credits = credits.rename(columns={"movie_id": "id"})
movies = movies.merge(credits, on='id')
movies_cleaned = movies.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
movies_cleaned['overview'] = movies_cleaned['overview'].fillna('')
bollywood['overview'] = bollywood['overview'].fillna('')
movies_cleaned = pd.concat([movies_cleaned, bollywood], ignore_index=True)
tfv = TfidfVectorizer(min_df=3, stop_words='english', ngram_range=(1, 3), token_pattern=r'\w{1,}')
tfv_matrix = tfv.fit_transform(movies_cleaned['overview'])
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()
def give_recommendations(title, sig=sig):
    idx = indices.get(title)
    if idx is None:
        return ["Movie not found in database."]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]
    movie_indices = [i[0] for i in sig_scores]
    return movies_cleaned['original_title'].iloc[movie_indices]
st.title("🎬 Movie Recommendation System")

movie_list = movies_cleaned['original_title'].dropna().unique()
selected_movie = st.selectbox("Select a movie", sorted(movie_list))

if st.button("Show Recommendations"):
    recommendations = give_recommendations(selected_movie)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write("✅", movie)
