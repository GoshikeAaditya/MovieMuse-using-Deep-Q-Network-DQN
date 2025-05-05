import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from data_preprocessing import DataPreprocessor
from rl_environment import MovieRecommendationEnv

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Create directories for models and results
os.makedirs('models/dqn', exist_ok=True)
os.makedirs('results/dqn', exist_ok=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=1e-4, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Create Q networks
        self.q_network = DQN(state_size, action_size, hidden_size).to(device)
        self.target_network = DQN(state_size, action_size, hidden_size).to(device)
        self.update_target_network()  # Copy weights to target network
        
        # Define optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Track training metrics
        self.loss_history = []
    
    def update_target_network(self):
        """Copy weights from Q network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions=None):
        """Choose an action using epsilon-greedy policy"""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
            
        if not valid_actions:
            return None
        
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Get action values from Q network
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        self.q_network.train()
        
        # Create a mask for valid actions
        mask = np.ones(self.action_size) * float('-inf')
        mask[valid_actions] = 0
        
        # Apply mask and get best action
        q_values = q_values + mask
        return np.argmax(q_values)
    
    def replay(self):
        """Train the network using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([experience[0] for experience in minibatch]).to(device)
        actions = torch.LongTensor([[experience[1]] for experience in minibatch]).to(device)
        rewards = torch.FloatTensor([[experience[2]] for experience in minibatch]).to(device)
        next_states = torch.FloatTensor([experience[3] for experience in minibatch]).to(device)
        dones = torch.FloatTensor([[experience[4]] for experience in minibatch]).to(device)
        
        # Compute Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
        
        # Compute target
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store loss
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def save(self, filepath):
        """Save the model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'loss_history': self.loss_history
        }, filepath)
    
    def load(self, filepath):
        """Load the model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.loss_history = checkpoint['loss_history']
            print(f"Model loaded from {filepath}")

# Load and preprocess data
print("Loading and preprocessing data...")
preprocessor = DataPreprocessor('Dataset/ratings.csv', 'Dataset/movies.csv')
preprocessor.load_data()
train_data, test_data, user_item_matrix = preprocessor.preprocess()
movie_features = preprocessor.get_movie_features()

# Create the environment
env = MovieRecommendationEnv(user_item_matrix, train_data, movie_features)

# Get state and action dimensions
state_size = env.state_size
action_size = env.n_movies

print(f"State size: {state_size}")
print(f"Action size (number of movies): {action_size}")

# Create the DQN agent
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    hidden_size=256,
    learning_rate=1e-4,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    memory_size=10000,
    batch_size=64
)

# Training parameters
episodes = 1000
target_update_frequency = 10  # Update target network every N episodes

# Initialize metrics tracking
rewards_history = []
avg_rewards_history = []
coverage_history = []  # Track movie coverage
exploration_rate_history = []  # Track exploration rate
diversity_history = []  # Track recommendation diversity
epsilon_history = []  # Track epsilon values
loss_history = []  # Track loss values

# Set to track all movies recommended across episodes
all_recommended_movies = set()

print("Starting training...")
for e in tqdm(range(episodes)):
    # Reset the environment
    user_id = np.random.randint(0, env.n_users)
    state = env.reset(user_id)
    
    total_reward = 0
    episode_recommendations = []
    episode_loss = []
    done = False
    
    while not done:
        # Get valid actions (movies not yet recommended)
        valid_actions = list(set(range(env.n_movies)) - env.already_recommended)
        
        # Choose an action
        action = agent.act(state, valid_actions)
        episode_recommendations.append(action)
        
        # Take the action
        next_state, reward, done, info = env.step(action)
        
        # Store experience in replay memory
        agent.remember(state, action, reward, next_state, done)
        
        # Train the network
        loss = agent.replay()
        if loss > 0:
            episode_loss.append(loss)
        
        # Update state and total reward
        state = next_state
        total_reward += reward
    
    # Update target network periodically
    if e % target_update_frequency == 0:
        agent.update_target_network()
    
    # Add recommended movies to the overall set
    all_recommended_movies.update(episode_recommendations)
    
    # Calculate coverage (percentage of all movies recommended so far)
    coverage = len(all_recommended_movies) / env.n_movies
    
    # Calculate exploration rate (percentage of new movies in this episode)
    new_recommendations = len(set(episode_recommendations) - set(list(all_recommended_movies)[:-len(episode_recommendations)]))
    exploration_rate = new_recommendations / len(episode_recommendations) if episode_recommendations else 0
    
    # Calculate diversity (using standard deviation of movie indices as a proxy)
    diversity = np.std(episode_recommendations) if episode_recommendations else 0
    
    # Save metrics history
    rewards_history.append(total_reward)
    avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
    avg_rewards_history.append(avg_reward)
    coverage_history.append(coverage)
    exploration_rate_history.append(exploration_rate)
    diversity_history.append(diversity)
    epsilon_history.append(agent.epsilon)
    
    # Save average loss for this episode
    avg_loss = np.mean(episode_loss) if episode_loss else 0
    loss_history.append(avg_loss)
    
    if e % 100 == 0:
        print(f"Episode: {e}, Avg Reward: {avg_reward:.2f}, Coverage: {coverage:.2f}, Epsilon: {agent.epsilon:.2f}")
        # Save the model
        agent.save(f"models/dqn/dqn_model_episode_{e}.pt")

# Save the final model
agent.save("models/dqn/dqn_model_final.pt")

# Create a comprehensive visualization dashboard
plt.figure(figsize=(15, 12))

# Plot 1: Rewards
plt.subplot(3, 3, 1)
plt.plot(rewards_history, alpha=0.3, color='blue', label='Rewards')
plt.plot(avg_rewards_history, color='red', label='Avg Rewards (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode')
plt.legend()

# Plot 2: Coverage
plt.subplot(3, 3, 2)
plt.plot(coverage_history, color='green')
plt.xlabel('Episode')
plt.ylabel('Coverage')
plt.title('Movie Coverage')

# Plot 3: Exploration Rate
plt.subplot(3, 3, 3)
plt.plot(exploration_rate_history, color='purple')
plt.xlabel('Episode')
plt.ylabel('Exploration Rate')
plt.title('Exploration Efficiency')

# Plot 4: Diversity
plt.subplot(3, 3, 4)
plt.plot(diversity_history, color='orange')
plt.xlabel('Episode')
plt.ylabel('Diversity')
plt.title('Recommendation Diversity')

# Plot 5: Epsilon Decay
plt.subplot(3, 3, 5)
plt.plot(epsilon_history, color='brown')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Exploration Parameter (Epsilon)')

# Plot 6: Loss
plt.subplot(3, 3, 6)
plt.plot(loss_history, color='magenta')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Training Loss')

# Plot 7: Final Metrics Bar Chart
plt.subplot(3, 3, 7)
final_metrics = [
    np.mean(rewards_history[-100:]),  # Final avg reward
    coverage_history[-1],  # Final coverage
    np.mean(exploration_rate_history[-100:]),  # Final avg exploration
    np.mean(diversity_history[-100:])  # Final avg diversity
]
labels = ['Reward', 'Coverage', 'Exploration', 'Diversity']
plt.bar(labels, final_metrics, color=['red', 'green', 'purple', 'orange'])
plt.ylabel('Value')
plt.title('Final Performance Metrics')

# Plot 8: Reward vs Epsilon
plt.subplot(3, 3, 8)
plt.scatter(epsilon_history, rewards_history, alpha=0.5, s=10)
plt.xlabel('Epsilon')
plt.ylabel('Reward')
plt.title('Reward vs Exploration Parameter')

# Plot 9: Reward vs Loss
plt.subplot(3, 3, 9)
plt.scatter(loss_history, rewards_history, alpha=0.5, s=10)
plt.xlabel('Loss')
plt.ylabel('Reward')
plt.title('Reward vs Training Loss')

plt.tight_layout()
plt.savefig('results/dqn/dqn_performance_dashboard.png')

# Also create individual high-resolution plots
# Rewards plot
plt.figure(figsize=(12, 6))
plt.plot(rewards_history, alpha=0.3, color='blue', label='Rewards')
plt.plot(avg_rewards_history, color='red', label='Avg Rewards (100 episodes)')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress')
plt.legend()
plt.savefig('results/dqn/dqn_training_progress.png')

# Coverage plot
plt.figure(figsize=(12, 6))
plt.plot(coverage_history, color='green')
plt.xlabel('Episode')
plt.ylabel('Coverage')
plt.title('Movie Coverage Over Time')
plt.savefig('results/dqn/dqn_coverage_progress.png')

# Exploration rate plot
plt.figure(figsize=(12, 6))
plt.plot(exploration_rate_history, color='purple')
plt.xlabel('Episode')
plt.ylabel('Exploration Rate')
plt.title('Exploration Efficiency Over Time')
plt.savefig('results/dqn/dqn_exploration_progress.png')

# Diversity plot
plt.figure(figsize=(12, 6))
plt.plot(diversity_history, color='orange')
plt.xlabel('Episode')
plt.ylabel('Diversity')
plt.title('Recommendation Diversity Over Time')
plt.savefig('results/dqn/dqn_diversity_progress.png')

# Epsilon decay plot
plt.figure(figsize=(12, 6))
plt.plot(epsilon_history, color='brown')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Exploration Parameter (Epsilon) Decay')
plt.savefig('results/dqn/dqn_epsilon_decay.png')

# Loss plot
plt.figure(figsize=(12, 6))
plt.plot(loss_history, color='magenta')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.savefig('results/dqn/dqn_loss.png')

plt.close('all')

print("Training completed!")
print("Performance dashboard saved to results/dqn/dqn_performance_dashboard.png")