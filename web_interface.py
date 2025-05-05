from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
import json

from data_preprocessing import DataPreprocessor
from rl_environment import MovieRecommendationEnv
from dqn_agent import DQNAgent

app = Flask(__name__)

# Global variables to store loaded models and data
preprocessor = None
env = None
agent = None
user_id_map = None
movie_id_map = None
movie_title_map = None

# In the load_models function
def load_models():
    """Load all necessary models and data"""
    global preprocessor, env, agent, user_id_map, movie_id_map, movie_title_map
    
    # Load and preprocess data with sampling for development
    sample_size = 100000  # Adjust based on your system's memory
    preprocessor = DataPreprocessor('Dataset/ratings.csv', 'Dataset/movies.csv', sample_size=sample_size)
    preprocessor.load_data()
    train_data, test_data, user_item_matrix = preprocessor.preprocess()
    movie_features = preprocessor.get_movie_features()
    
    # User features completely removed (not just disabled)
    user_features = None
    
    # Get ID mappings
    user_id_map, movie_id_map, movie_title_map = preprocessor.get_mappings()
    
    # Create the environment (only once)
    env = MovieRecommendationEnv(user_item_matrix, train_data, movie_features, user_features)
    
    # Get state and action dimensions
    state_size = env.get_state().shape[0]
    action_size = env.n_movies
    
    # Create and load the agent
    agent = DQNAgent(state_size, action_size)
    model_path = 'models/dqn_model_final.h5'
    if os.path.exists(model_path):
        agent.load(model_path)
    agent.epsilon = 0.01  # Set low epsilon for exploitation

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/users')
def get_users():
    """Return a list of users"""
    if user_id_map is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    users = [{"id": int(original_id), "encoded_id": int(encoded_id)} 
             for encoded_id, original_id in user_id_map.items()]
    return jsonify(users)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Generate recommendations for a user"""
    if env is None or agent is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    data = request.json
    user_id = data.get('user_id')
    top_k = data.get('top_k', 10)
    
    if user_id is None:
        return jsonify({"error": "User ID is required"}), 400
    
    # Convert to encoded user ID if necessary
    encoded_user_id = user_id
    if user_id in user_id_map.values():
        # Find the encoded ID
        encoded_user_id = [k for k, v in user_id_map.items() if v == user_id][0]
    
    # Get recommendations
    state = env.reset(encoded_user_id)
    recommendations = []
    
    for _ in range(top_k):
        valid_actions = list(set(range(env.n_movies)) - env.already_recommended)
        action = agent.act(state, valid_actions)
        
        # Get movie details
        original_movie_id = movie_id_map.get(action, None)
        if original_movie_id is not None:
            title = movie_title_map.get(original_movie_id, f"Unknown Movie {original_movie_id}")
            recommendations.append({
                "movie_id": int(original_movie_id),
                "encoded_id": int(action),
                "title": title
            })
        
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break
    
    return jsonify(recommendations)

@app.route('/movie/<int:movie_id>')
def get_movie(movie_id):
    """Get details for a specific movie"""
    if preprocessor is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    # Find the movie in the dataset
    movie_data = preprocessor.movies_df[preprocessor.movies_df['movieId'] == movie_id]
    
    if movie_data.empty:
        return jsonify({"error": "Movie not found"}), 404
    
    # Get movie details
    movie = movie_data.iloc[0]
    
    # Get average rating
    ratings = preprocessor.ratings_df[preprocessor.ratings_df['movieId'] == movie_id]
    avg_rating = ratings['rating'].mean() if not ratings.empty else 0
    
    return jsonify({
        "id": int(movie['movieId']),
        "title": movie['title'],
        "genres": movie['genres'].split('|'),
        "average_rating": float(avg_rating),
        "rating_count": int(len(ratings))
    })

@app.route('/search', methods=['POST'])
def search_movies():
    """Search for movies by title or genre"""
    if preprocessor is None:
        print("Error: Models not loaded")
        return jsonify({"error": "Models not loaded"}), 500
    
    data = request.json
    print(f"Search request received: {data}")
    
    query = data.get('query', '').lower()
    search_type = data.get('type', 'title')  # 'title' or 'genre'
    
    print(f"Searching for '{query}' by {search_type}")
    
    if not query:
        print("Error: Empty search query")
        return jsonify({"error": "Search query is required"}), 400
    
    results = []
    
    try:
        # Print some info about the movies dataframe
        print(f"Movies dataframe shape: {preprocessor.movies_df.shape}")
        print(f"Sample movie titles: {preprocessor.movies_df['title'].head().tolist()}")
        
        if search_type == 'title':
            # Case-insensitive search by title
            matching_movies = preprocessor.movies_df[
                preprocessor.movies_df['title'].str.lower().str.contains(query, case=False, regex=True)
            ]
            print(f"Found {len(matching_movies)} movies matching title '{query}'")
        elif search_type == 'genre':
            # Case-insensitive search by genre
            matching_movies = preprocessor.movies_df[
                preprocessor.movies_df['genres'].str.lower().str.contains(query, case=False, regex=True)
            ]
            print(f"Found {len(matching_movies)} movies matching genre '{query}'")
        else:
            print(f"Error: Invalid search type '{search_type}'")
            return jsonify({"error": "Invalid search type"}), 400
        
        # Get ratings information for matching movies
        for _, movie in matching_movies.iterrows():
            movie_id = movie['movieId']
            ratings = preprocessor.ratings_df[preprocessor.ratings_df['movieId'] == movie_id]
            avg_rating = ratings['rating'].mean() if not ratings.empty else 0
            
            results.append({
                "id": int(movie_id),
                "title": movie['title'],
                "genres": movie['genres'].split('|'),
                "average_rating": float(avg_rating) if not np.isnan(avg_rating) else 0,
                "rating_count": int(len(ratings))
            })
        
        # Sort results by average rating (descending)
        results = sorted(results, key=lambda x: x['average_rating'], reverse=True)
        
        print(f"Search for '{query}' returned {len(results)} results")
        
        return jsonify(results)
    except Exception as e:
        import traceback
        print(f"Error in search: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route('/recommend-similar', methods=['POST'])
def recommend_similar():
    """Recommend movies similar to a given movie"""
    if env is None or agent is None or preprocessor is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    data = request.json
    movie_id = data.get('movie_id')
    top_k = data.get('top_k', 10)
    
    if movie_id is None:
        return jsonify({"error": "Movie ID is required"}), 400
    
    try:
        # Find the movie in the dataset
        movie_data = preprocessor.movies_df[preprocessor.movies_df['movieId'] == movie_id]
        
        if movie_data.empty:
            return jsonify({"error": "Movie not found"}), 404
        
        # Get the movie's genres
        genres = movie_data.iloc[0]['genres'].split('|')
        
        # Find movies with similar genres
        similar_movies = []
        
        for _, movie in preprocessor.movies_df.iterrows():
            if movie['movieId'] == movie_id:
                continue  # Skip the input movie
            
            movie_genres = movie['genres'].split('|')
            # Calculate genre similarity (intersection / union)
            common_genres = set(genres).intersection(set(movie_genres))
            all_genres = set(genres).union(set(movie_genres))
            similarity = len(common_genres) / len(all_genres) if all_genres else 0
            
            if similarity > 0:
                similar_movies.append((movie, similarity))
        
        # Sort by similarity
        similar_movies.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k similar movies
        recommendations = []
        for movie, similarity in similar_movies[:top_k]:
            movie_id = movie['movieId']
            ratings = preprocessor.ratings_df[preprocessor.ratings_df['movieId'] == movie_id]
            avg_rating = ratings['rating'].mean() if not ratings.empty else 0
            
            recommendations.append({
                "id": int(movie_id),
                "title": movie['title'],
                "genres": movie['genres'].split('|'),
                "average_rating": float(avg_rating) if not np.isnan(avg_rating) else 0,
                "rating_count": int(len(ratings)),
                "similarity": float(similarity)
            })
        
        print(f"Found {len(recommendations)} similar movies")
        
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in recommend-similar: {str(e)}")
        return jsonify({"error": f"Recommendation failed: {str(e)}"}), 500

# Add this route to check if the dataset is loaded correctly
@app.route('/dataset-info')
def dataset_info():
    """Return information about the loaded dataset"""
    if preprocessor is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    try:
        # Get basic dataset info
        movie_count = len(preprocessor.movies_df)
        user_count = len(preprocessor.ratings_df['userId'].unique())
        rating_count = len(preprocessor.ratings_df)
        
        # Get sample movie titles
        sample_movies = preprocessor.movies_df['title'].head(20).tolist()
        
        # Get sample genres
        all_genres = set()
        for genres in preprocessor.movies_df['genres']:
            all_genres.update(genres.split('|'))
        
        return jsonify({
            "movie_count": movie_count,
            "user_count": user_count,
            "rating_count": rating_count,
            "sample_movies": sample_movies,
            "genres": sorted(list(all_genres))
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get dataset info: {str(e)}"}), 500

@app.route('/recommend-content-based', methods=['POST'])
def recommend_content_based():
    """Recommend movies based on content (genres, keywords, etc.)"""
    if preprocessor is None:
        return jsonify({"error": "Models not loaded"}), 500
    
    data = request.json
    movie_id = data.get('movie_id')
    top_k = data.get('top_k', 10)
    
    if movie_id is None:
        return jsonify({"error": "Movie ID is required"}), 400
    
    try:
        # Find the movie in the dataset
        movie_data = preprocessor.movies_df[preprocessor.movies_df['movieId'] == movie_id]
        
        if movie_data.empty:
            return jsonify({"error": "Movie not found"}), 404
        
        # Get content-based recommendations
        recommendations = []
        
        # Get the movie's genres and other features
        target_movie = movie_data.iloc[0]
        target_genres = set(target_movie['genres'].split('|'))
        
        # Calculate similarity scores for all movies
        similarity_scores = []
        
        for _, movie in preprocessor.movies_df.iterrows():
            if movie['movieId'] == movie_id:
                continue  # Skip the input movie
            
            # Calculate genre similarity
            movie_genres = set(movie['genres'].split('|'))
            genre_similarity = len(target_genres.intersection(movie_genres)) / len(target_genres.union(movie_genres)) if movie_genres else 0
            
            # You can add more similarity factors here (actors, directors, keywords, etc.)
            
            similarity_scores.append((movie, genre_similarity))
        
        # Sort by similarity score
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k recommendations
        for movie, similarity in similarity_scores[:top_k]:
            movie_id = movie['movieId']
            ratings = preprocessor.ratings_df[preprocessor.ratings_df['movieId'] == movie_id]
            avg_rating = ratings['rating'].mean() if not ratings.empty else 0
            
            recommendations.append({
                "id": int(movie_id),
                "title": movie['title'],
                "genres": movie['genres'].split('|'),
                "average_rating": float(avg_rating) if not np.isnan(avg_rating) else 0,
                "rating_count": int(len(ratings)),
                "similarity": float(similarity)
            })
        
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in content-based recommendation: {str(e)}")
        return jsonify({"error": f"Recommendation failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Load models before starting the server
    load_models()
    # Use a different port, e.g., 5002
    app.run(debug=True, port=5002)