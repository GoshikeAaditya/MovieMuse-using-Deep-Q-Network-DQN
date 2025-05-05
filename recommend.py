import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from data_preprocessing import DataPreprocessor
from rl_environment import MovieRecommendationEnv
from dqn_agent import DQNAgent

def evaluate_agent(agent, env, test_data, n_users=100, top_k=10):
    """Evaluate the agent on test data"""
    precision_at_k = []
    recall_at_k = []
    ndcg_at_k = []
    accuracy_at_k = []
    
    # Sample users for evaluation
    test_users = np.random.choice(test_data['userIdEncoded'].unique(), 
                                 size=min(n_users, len(test_data['userIdEncoded'].unique())), 
                                 replace=False)
    
    for user_id in test_users:
        # Get user's actual ratings from test data
        user_test_data = test_data[test_data['userIdEncoded'] == user_id]
        actual_liked_movies = set(user_test_data[user_test_data['rating'] >= 4]['movieIdEncoded'].values)
        
        if len(actual_liked_movies) == 0:
            continue
        
        # Reset environment for this user
        state = env.reset(user_id)
        
        # Get recommendations
        recommendations = []
        for _ in range(top_k):
            valid_actions = list(set(range(env.n_movies)) - env.already_recommended)
            action = agent.act(state, valid_actions)
            recommendations.append(action)
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                break
        
        # Calculate precision and recall
        recommended_and_liked = len(set(recommendations) & actual_liked_movies)
        precision = recommended_and_liked / len(recommendations) if recommendations else 0
        recall = recommended_and_liked / len(actual_liked_movies) if actual_liked_movies else 0
        
        precision_at_k.append(precision)
        recall_at_k.append(recall)
        
        # Calculate accuracy (proportion of recommendations that were liked)
        accuracy = recommended_and_liked / len(recommendations) if recommendations else 0
        accuracy_at_k.append(accuracy)
        
        # Calculate NDCG
        dcg = 0
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual_liked_movies), top_k))])
        
        for i, movie_id in enumerate(recommendations):
            if movie_id in actual_liked_movies:
                dcg += 1 / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_at_k.append(ndcg)
    
    return {
        'precision': np.mean(precision_at_k),
        'recall': np.mean(recall_at_k),
        'ndcg': np.mean(ndcg_at_k),
        'accuracy': np.mean(accuracy_at_k)
    }

def get_recommendations(agent, env, user_id, n_recommendations=10):
    """Get movie recommendations for a specific user"""
    state = env.reset(user_id)
    
    recommendations = []
    for _ in range(n_recommendations):
        valid_actions = list(set(range(env.n_movies)) - env.already_recommended)
        action = agent.act(state, valid_actions)
        recommendations.append(action)
        next_state, _, done, _ = env.step(action)
        state = next_state
        if done:
            break
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--user_id', type=int, help='User ID for recommendations')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--model_path', type=str, default='models/dqn_model_final.h5', 
                        help='Path to the trained model')
    parser.add_argument('--top_k', type=int, default=10, 
                        help='Number of recommendations to generate')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor('Dataset/ratings.csv', 'Dataset/movies.csv')
    preprocessor.load_data()
    train_data, test_data, user_item_matrix = preprocessor.preprocess()
    movie_features = preprocessor.get_movie_features()
    
    # Get ID mappings
    user_id_map, movie_id_map, movie_title_map = preprocessor.get_mappings()
    
    # Create the environment
    env = MovieRecommendationEnv(user_item_matrix, train_data, movie_features)
    
    # Get state and action dimensions
    state_size = env.get_state().shape[0]
    action_size = env.n_movies
    
    # Create and load the agent
    agent = DQNAgent(state_size, action_size)
    agent.load(args.model_path)
    agent.epsilon = 0.01  # Set low epsilon for exploitation
    
    if args.evaluate:
        # Evaluate the model
        print("Evaluating the model...")
        metrics = evaluate_agent(agent, env, test_data, top_k=args.top_k)
        print(f"Precision@{args.top_k}: {metrics['precision']:.4f}")
        print(f"Recall@{args.top_k}: {metrics['recall']:.4f}")
        print(f"NDCG@{args.top_k}: {metrics['ndcg']:.4f}")
        print(f"Accuracy@{args.top_k}: {metrics['accuracy']:.4f}")
    
    if args.user_id is not None:
        # Convert to encoded user ID if necessary
        encoded_user_id = args.user_id
        if args.user_id in user_id_map.values():
            # Find the encoded ID
            encoded_user_id = [k for k, v in user_id_map.items() if v == args.user_id][0]
        
        # Get recommendations
        print(f"Generating recommendations for user {args.user_id}...")
        movie_ids = get_recommendations(agent, env, encoded_user_id, args.top_k)
        
        # Convert to original movie IDs and titles
        recommendations = []
        for movie_id in movie_ids:
            original_movie_id = movie_id_map.get(movie_id, None)
            if original_movie_id is not None:
                title = movie_title_map.get(original_movie_id, f"Unknown Movie {original_movie_id}")
                recommendations.append((original_movie_id, title))
        
        # Print recommendations
        print("\nRecommended Movies:")
        for i, (movie_id, title) in enumerate(recommendations, 1):
            print(f"{i}. {title} (ID: {movie_id})")

if __name__ == "__main__":
    main()