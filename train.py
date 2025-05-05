import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from data_preprocessing import DataPreprocessor
from rl_environment import MovieRecommendationEnv
from dqn_agent import DQNAgent

# Set random seeds for reproducibility
np.random.seed(42)

# Create directories for models and results
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load and preprocess data
print("Loading and preprocessing data...")
preprocessor = DataPreprocessor('Dataset/ratings.csv', 'Dataset/movies.csv')
preprocessor.load_data()
train_data, test_data, user_item_matrix = preprocessor.preprocess()
movie_features = preprocessor.get_movie_features()

# Create the environment
env = MovieRecommendationEnv(user_item_matrix, train_data, movie_features)

# Get state and action dimensions
state_size = env.get_state().shape[0]
action_size = env.n_movies

print(f"State size: {state_size}")
print(f"Action size: {action_size}")

# Create the agent
agent = DQNAgent(state_size, action_size)

# Training parameters
episodes = 1000
batch_size = 32
update_target_every = 10

# Training loop
rewards_history = []
avg_rewards_history = []

print("Starting training...")
for e in tqdm(range(episodes)):
    # Reset the environment
    user_id = np.random.randint(0, env.n_users)
    state = env.reset(user_id)
    
    total_reward = 0
    done = False
    
    while not done:
        # Get valid actions (movies not yet recommended)
        valid_actions = list(set(range(env.n_movies)) - env.already_recommended)
        
        # Choose an action
        action = agent.act(state, valid_actions)
        
        # Take the action
        next_state, reward, done, info = env.step(action)
        
        # Remember the experience
        agent.remember(state, action, reward, next_state, done)
        
        # Update state and total reward
        state = next_state
        total_reward += reward
        
        # Train the agent
        agent.replay(batch_size)
    
    # Update target model periodically
    if e % update_target_every == 0:
        agent.update_target_model()
    
    # Save rewards history
    rewards_history.append(total_reward)
    avg_reward = np.mean(rewards_history[-100:])
    avg_rewards_history.append(avg_reward)
    
    if e % 100 == 0:
        print(f"Episode: {e}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        # Save the model
        agent.save(f"models/dqn_model_episode_{e}.h5")

# Save the final model
agent.save("models/dqn_model_final.h5")

# Plot training progress
plt.figure(figsize=(12, 6))
plt.plot(rewards_history, alpha=0.3, color='blue', label='Rewards')
plt.plot(avg_rewards_history, color='red', label='Avg Rewards (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.legend()
plt.savefig('results/training_progress.png')
plt.close()

print("Training completed!")