import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

class DataPreprocessor:
    def __init__(self, ratings_path, movies_path, sample_size=None):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.sample_size = sample_size
        
    def load_data(self):
        """Load ratings and movies data"""
        # Load data
        self.ratings_df = pd.read_csv(self.ratings_path)
        self.movies_df = pd.read_csv(self.movies_path)
        
        # Sample data if needed (for development/testing)
        if self.sample_size and self.sample_size < len(self.ratings_df):
            self.ratings_df = self.ratings_df.sample(n=self.sample_size, random_state=42)
        
        print(f"Loaded {len(self.ratings_df)} ratings and {len(self.movies_df)} movies")
        
        # Display sample data
        print("\nRatings sample:")
        print(self.ratings_df.head())
        print("\nMovies sample:")
        print(self.movies_df.head())
        
    def preprocess(self):
        """Preprocess the data for the RL environment"""
        # Encode user and movie IDs
        self.ratings_df['userIdEncoded'] = self.user_encoder.fit_transform(self.ratings_df['userId'])
        self.ratings_df['movieIdEncoded'] = self.movie_encoder.fit_transform(self.ratings_df['movieId'])
        
        # Get number of users and movies
        self.n_users = len(self.ratings_df['userIdEncoded'].unique())
        self.n_movies = len(self.ratings_df['movieIdEncoded'].unique())
        
        print(f"Number of unique users: {self.n_users}")
        print(f"Number of unique movies: {self.n_movies}")
        
        # Create sparse user-item matrix instead of dense pivot table
        row = self.ratings_df['userIdEncoded'].values
        col = self.ratings_df['movieIdEncoded'].values
        data = self.ratings_df['rating'].values
        
        self.user_item_matrix = csr_matrix((data, (row, col)), shape=(self.n_users, self.n_movies))
        
        # Split data into train and test
        self.train_data, self.test_data = train_test_split(
            self.ratings_df, test_size=0.2, random_state=42
        )
        
        print(f"Training data size: {len(self.train_data)}")
        print(f"Testing data size: {len(self.test_data)}")
        
        return self.train_data, self.test_data, self.user_item_matrix
    
    def get_movie_features(self):
        """Extract movie features from genres"""
        # One-hot encode genres
        genres = self.movies_df['genres'].str.get_dummies('|')
        
        # Merge with movie IDs
        movie_features = pd.concat([self.movies_df[['movieId']], genres], axis=1)
        
        # Map to encoded movie IDs
        id_mapping = dict(zip(self.ratings_df['movieId'], self.ratings_df['movieIdEncoded']))
        movie_features['movieIdEncoded'] = movie_features['movieId'].map(id_mapping)
        
        # Drop rows with NaN (movies not in ratings)
        movie_features = movie_features.dropna(subset=['movieIdEncoded'])
        movie_features['movieIdEncoded'] = movie_features['movieIdEncoded'].astype(int)
        movie_features = movie_features.set_index('movieIdEncoded')
        
        # Drop original movieId
        movie_features = movie_features.drop('movieId', axis=1)
        
        return movie_features
    
    def get_mappings(self):
        """Return ID mappings for converting between original and encoded IDs"""
        user_id_map = dict(zip(self.ratings_df['userIdEncoded'], self.ratings_df['userId']))
        movie_id_map = dict(zip(self.ratings_df['movieIdEncoded'], self.ratings_df['movieId']))
        movie_title_map = dict(zip(self.movies_df['movieId'], self.movies_df['title']))
        
        return user_id_map, movie_id_map, movie_title_map
    
    def extract_user_features(self):
        """Extract user features from rating patterns"""
        user_features = {}
        
        # Group by user
        user_groups = self.ratings_df.groupby('userIdEncoded')
        
        for user_id, group in user_groups:
            # Calculate basic statistics
            avg_rating = group['rating'].mean()
            rating_count = len(group)
            rating_std = group['rating'].std()
            
            # Calculate genre preferences
            if hasattr(self, 'movies_df') and 'genres' in self.movies_df.columns:
                user_movies = group['movieId'].values
                user_movie_genres = self.movies_df[self.movies_df['movieId'].isin(user_movies)]
                
                # Extract all genres
                all_genres = []
                for genres in user_movie_genres['genres'].values:
                    all_genres.extend(genres.split('|'))
                
                # Count genre occurrences
                genre_counts = {}
                for genre in set(all_genres):
                    genre_counts[genre] = all_genres.count(genre) / rating_count
            
            # Store user features
            user_features[user_id] = {
                'avg_rating': avg_rating,
                'rating_count': rating_count,
                'rating_std': rating_std if not np.isnan(rating_std) else 0,
                'genre_preferences': genre_counts if 'genre_counts' in locals() else {}
            }
        
        return user_features
    
    def get_user_features_matrix(self):
        """Convert user features to a matrix format"""
        user_features = self.extract_user_features()
        
        # Get all possible genres
        all_genres = set()
        for user_data in user_features.values():
            all_genres.update(user_data['genre_preferences'].keys())
        
        # Create feature matrix
        feature_columns = ['avg_rating', 'rating_count', 'rating_std'] + list(all_genres)
        user_feature_matrix = np.zeros((self.n_users, len(feature_columns)))
        
        for user_id, features in user_features.items():
            user_feature_matrix[user_id, 0] = features['avg_rating']
            user_feature_matrix[user_id, 1] = features['rating_count'] / self.n_movies  # Normalize
            user_feature_matrix[user_id, 2] = features['rating_std']
            
            # Add genre preferences
            for i, genre in enumerate(all_genres):
                user_feature_matrix[user_id, 3 + i] = features['genre_preferences'].get(genre, 0)
        
        return user_feature_matrix, feature_columns