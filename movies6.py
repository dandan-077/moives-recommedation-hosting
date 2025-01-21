import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import os

# Load movie dataset
movies = pd.read_csv("tmdb_movies_data.csv")  # Update path to your dataset

# Preprocess the data
movies['genres'] = movies['genres'].fillna('').apply(lambda x: x.replace(' ', '').replace('|', ' '))
movies['cast'] = movies['cast'].fillna('').apply(lambda x: ' '.join(x.split(',')[:3]))  # Top 3 actors
movies['director'] = movies['director'].fillna('')
movies['keywords'] = movies['keywords'].fillna('')
movies['combined_features'] = (movies['genres'] + " " + movies['cast'] + " " + movies['director'] + " " + movies['keywords'])

# CountVectorizer for feature extraction
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(movies['combined_features'])

# TMDb API configuration
TMDB_API_KEY = "3c56e04d763dc09cecbc7424e0e65636"  # Replace with your actual API key
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

# Fetch movie details from TMDb
def get_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url)
        data = response.json()

        poster_url = TMDB_IMAGE_URL + data.get('poster_path', '') if data.get('poster_path') else None
        genres = ", ".join([genre['name'] for genre in data.get('genres', [])])
        summary = data.get('overview', 'No summary available.')

        # Fetch top cast
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
        credits_response = requests.get(credits_url)
        credits_data = credits_response.json()
        top_cast = [
            {
                'name': actor['name'],
                'photo': TMDB_IMAGE_URL + actor['profile_path'] if actor.get('profile_path') else None
            }
            for actor in credits_data.get('cast', [])[:3]
        ]

        return poster_url, genres, summary, top_cast
    except Exception as e:
        print(f"Error fetching movie details: {e}")
        return None, "", "No details available.", []

# Movie recommendation function
def get_recommendations(user_preferences, num_recommendations=5):
    user_features = user_preferences['genres'] + " " + user_preferences['actors'] + " " + user_preferences['director']
    user_vector = cv.transform([user_features])
    cosine_sim = cosine_similarity(user_vector, count_matrix)
    similarity_scores = list(enumerate(cosine_sim[0]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_movies_indices = [i[0] for i in sorted_scores[:num_recommendations]]
    return movies.iloc[top_movies_indices]

# Simulated user database
USER_DB = 'users.json'

def register_user(username, password, preferences):
    if not os.path.exists(USER_DB):
        with open(USER_DB, 'w') as f:
            json.dump({}, f)
    with open(USER_DB, 'r') as f:
        users = json.load(f)
    users[username] = {'password': password, 'preferences': preferences}
    with open(USER_DB, 'w') as f:
        json.dump(users, f)

def check_user(username, password):
    if not os.path.exists(USER_DB):
        return False
    with open(USER_DB, 'r') as f:
        users = json.load(f)
    return users.get(username) and users[username]['password'] == password

# Display movie details
def display_movie_details(movie_id, title):
    st.markdown(f"### {title}")
    poster_url, genres, summary, top_cast = get_movie_details(movie_id)

    if poster_url:
        st.image(poster_url, width=200)
    st.write(f"**Genres:** {genres}")
    st.write(f"**Summary:** {summary}")

    if top_cast:
        st.markdown("#### Top Cast")
        cols = st.columns(len(top_cast))
        for col, actor in zip(cols, top_cast):
            if actor['photo']:
                col.image(actor['photo'], width=100)
            col.caption(actor['name'])
    else:
        st.write("No cast information available.")
    st.divider()

# Streamlit UI
st.title("Movie Recommendation System")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    option = st.selectbox("Choose an option", ["Login", "Register"])

    if option == "Register":
        username = st.text_input("Enter Username")
        password = st.text_input("Enter Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        if password == confirm_password:
            preferences = {'genres': '', 'actors': '', 'director': ''}
            if st.button("Register"):
                register_user(username, password, preferences)
                st.success("Account created successfully!")
        else:
            st.error("Passwords do not match.")
    elif option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
            else:
                st.error("Invalid credentials.")

if st.session_state.logged_in:
    user = st.session_state.username
    with open(USER_DB, 'r') as f:
        users = json.load(f)
    user_preferences = users[user]['preferences']

    search_query = st.text_input("Search for a movie by name", key="search")
    if search_query:
        search_results = movies[movies['original_title'].str.contains(search_query, case=False, na=False)]
        if not search_results.empty:
            st.write("### Search Results:")
            for _, row in search_results.iterrows():
                display_movie_details(row['id'], row['original_title'])
        else:
            st.write("No results found.")

    show_preferences = st.checkbox("Update Preferences")
    if show_preferences:
        genres = st.multiselect("Select Genres", options= ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'])
        actors = st.text_input("Enter Favorite Actors (comma-separated)")
        director = st.text_input("Enter Favorite Director")
        if st.button("Update Preferences"):
            user_preferences = {'genres': ' '.join(genres), 'actors': actors, 'director': director}
            users[user]['preferences'] = user_preferences
            with open(USER_DB, 'w') as f:
                json.dump(users, f)
            st.success("Preferences updated!")

    st.write("### Movie Recommendations:")
    num_movies_to_show = st.session_state.get('num_movies', 5)
    recommended_movies = get_recommendations(user_preferences, num_recommendations=num_movies_to_show)

    for _, row in recommended_movies.iterrows():
        display_movie_details(row['id'], row['original_title'])

    if st.button("Load More"):
        st.session_state.num_movies = num_movies_to_show + 5
        st.experimental_rerun()

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()
