import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

# Load MovieLens dataset
@st.cache_data
def load_data():
    movies = pd.read_csv('data/movies.csv')  # Load your movies dataset
    ratings = pd.read_csv('data/ratings.csv')  # Load your ratings dataset
    return movies, ratings

movies, ratings = load_data()

# Movie preferences input
st.title("Movie Recommendation System")
st.header("Input your preferences")

genre = st.selectbox('Select your favorite genre', movies['genres'].unique())
movie_input = st.text_input('Enter a movie you like')

# Collaborative filtering-based recommendation
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)

model = SVD()
model.fit(trainset)

def recommend_movies(movie_name, genre):
    # Find movieId based on user input
    movie_id = movies[movies['title'].str.contains(movie_name, na=False)]['movieId'].values[0]
    
    # Predict ratings for all movies
    movie_list = movies[movies['genres'].str.contains(genre, na=False)]
    movie_list['est'] = movie_list['movieId'].apply(lambda x: model.predict(1, x).est)
    
    # Get top recommendations
    recommendations = movie_list.sort_values('est', ascending=False).head(10)
    return recommendations

if st.button('Recommend'):
    if movie_input:
        recommendations = recommend_movies(movie_input, genre)
        st.write("We recommend:")
        for index, row in recommendations.iterrows():
            st.write(f"{row['title']} - Predicted Rating: {row['est']:.2f}")
    else:
        st.write("Please enter a movie you like.")
