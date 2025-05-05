import numpy as np

def calculate_accuracy(recommendations, actual_liked_movies):
    """
    Calculate accuracy of recommendations
    
    Args:
        recommendations: List of recommended movie IDs
        actual_liked_movies: Set of movie IDs that the user actually liked
        
    Returns:
        accuracy: Proportion of recommended movies that the user actually liked
    """
    if not recommendations:
        return 0.0
    
    # Count how many recommended movies were actually liked by the user
    hits = sum(1 for movie_id in recommendations if movie_id in actual_liked_movies)
    
    # Calculate accuracy as the proportion of hits
    accuracy = hits / len(recommendations)
    
    return accuracy

def evaluate_accuracy(agent, env, test_data, n_users=100, top_k=10):
    """
    Evaluate the accuracy of the recommendation system
    
    Args:
        agent: The DQN agent
        env: The recommendation environment
        test_data: Test dataset
        n_users: Number of users to evaluate
        top_k: Number of recommendations to generate
        
    Returns:
        average_accuracy: Average accuracy across all evaluated users
    """
    accuracy_scores = []
    
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
        
        # Calculate accuracy for this user
        user_accuracy = calculate_accuracy(recommendations, actual_liked_movies)
        accuracy_scores.append(user_accuracy)
    
    # Return average accuracy across all users
    return np.mean(accuracy_scores) if accuracy_scores else 0.0