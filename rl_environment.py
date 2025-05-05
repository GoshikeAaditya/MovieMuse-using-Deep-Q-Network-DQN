import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class MovieRecommendationEnv:
    """
    Reinforcement Learning environment for movie recommendations
    
    State: User's rating history (vector of ratings for each movie)
    Action: Recommend a movie (movie index)
    Reward: Rating the user gives to the recommended movie
    """
    
    def __init__(self, user_item_matrix, train_data, movie_features=None, user_features=None):
        self.user_item_matrix = user_item_matrix
        self.train_data = train_data
        self.movie_features = movie_features
        # User features removed from consideration
        self.user_features = None
        
        self.n_users = user_item_matrix.shape[0]
        self.n_movies = user_item_matrix.shape[1]
        
        self.current_user = None
        self.user_state = None
        self.available_movies = None
        self.already_recommended = None
        
    def reset(self, user_id=None):
        """Reset the environment for a new episode"""
        # Select a random user if not specified
        if user_id is None:
            self.current_user = np.random.randint(0, self.n_users)
        else:
            self.current_user = user_id
            
        # Get user's rating history as the state
        if isinstance(self.user_item_matrix, csr_matrix):
            # For sparse matrix
            self.user_state = self.user_item_matrix[self.current_user].toarray().flatten()
        else:
            # For pandas DataFrame
            self.user_state = self.user_item_matrix.iloc[self.current_user].values
        
        # Track which movies have already been recommended
        self.already_recommended = set()
        
        # Get available movies (those with ratings > 0 in training data)
        user_ratings = self.train_data[self.train_data['userIdEncoded'] == self.current_user]
        self.available_movies = set(user_ratings['movieIdEncoded'].values)
        
        return self.get_state()
    
    def get_state(self):
        """Return a simplified state representation"""
        # Just return the user's rating history
        if self.user_state is None:
            return np.zeros(self.n_movies)
        return self.user_state
    
    def step(self, action):
        """
        Take an action (recommend a movie) and return the new state, reward, and done flag
        
        Args:
            action: Index of the movie to recommend
            
        Returns:
            next_state: New state after taking the action
            reward: Reward received
            done: Whether the episode is done
            info: Additional information
        """
        # Check if action is valid
        if action in self.already_recommended:
            # Penalize recommending the same movie twice
            return self.get_state(), -1, False, {"message": "Movie already recommended"}
        
        # Add to recommended set
        self.already_recommended.add(action)
        
        # Get the true rating from the user-item matrix
        true_rating = self.user_state[action]
        
        # Calculate reward based on the rating
        if true_rating == 0:  # User hasn't rated this movie
            reward = 0
        else:
            # Scale the reward: higher ratings give higher rewards
            reward = true_rating - 2.5  # Center around 0 (assuming ratings are 1-5)
        
        # Check if episode is done (recommended enough movies)
        done = len(self.already_recommended) >= min(10, len(self.available_movies))
        
        return self.get_state(), reward, done, {"rating": true_rating}