import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#data set
movies_data = {
    'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'Title': [
        'The Matrix', 'Inception', 'Titanic', 'The Dark Knight', 'Interstellar',
        'The Notebook', 'The Avengers', 'Forrest Gump', 'Gladiator', 'Avatar',
        'The Shawshank Redemption', 'Pulp Fiction', 'The Godfather', 'Toy Story', 'Frozen'
    ],
    'Genres': [
        'Action | Sci-Fi', 'Action | Sci-Fi | Thriller', 'Romance | Drama | Love',
        'Action | Crime | Drama', 'Adventure | Drama | Sci-Fi', 'Romance | Drama | Love',
        'Action | Adventure | Sci-Fi', 'Drama | Romance | Love', 'Action | Drama',
        'Action | Adventure | Fantasy', 'Drama', 'Crime | Drama', 'Crime | Drama',
        'Animation | Adventure | Comedy', 'Animation | Family | Fantasy'
    ]
}

# Create DataFrame
movies_df = pd.DataFrame(movies_data)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies_df['Genres'])
def recommendations_genres(user_genres, movies_df, tfidf, tfidf_matrix, top_n=3):
    user_tfidf = tfidf.transform([user_genres])
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[:top_n]]
    return movies_df['Title'].iloc[movie_indices].tolist()
print("Enter your preferred genres (e.g., 'Action Sci-Fi Love',"
      " separate multiple genres with spaces):")
user_genres = input().strip()
red_movies = recommendations_genres(user_genres, movies_df, tfidf, tfidf_matrix)
print(f"\nRecommendations based on genres '{user_genres}':")
for i, movie in enumerate(red_movies, 1):
    print(f"{i}. {movie}")