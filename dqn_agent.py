import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Memory for experience replay
        self.memory = deque(maxlen=2000)
        
        # Build the model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build a neural network model for DQN"""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target model with weights from the main model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions=None):
        """Choose an action based on the current state"""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
            
        if np.random.rand() <= self.epsilon:
            # Explore: choose a random action from valid actions
            return random.choice(valid_actions)
        
        # Exploit: predict Q-values and choose the best valid action
        act_values = self.model.predict(np.array([state]), verbose=0)[0]
        
        # Filter for valid actions only
        valid_act_values = [(i, act_values[i]) for i in valid_actions]
        return max(valid_act_values, key=lambda x: x[1])[0]
    
    def replay(self, batch_size):
        """Train the model with experiences from memory"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(np.array([next_state]), verbose=0)[0]
                )
            
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load the model weights from a file"""
        try:
            self.model.load_weights(name)
            print(f"Successfully loaded model from {name}")
        except (OSError, IOError) as e:
            print(f"Error loading model: {e}")
            print("If you haven't trained the model yet, please run 'python main.py --train' first.")
            # You might want to exit here or continue with untrained model
            # import sys
            # sys.exit(1)
    
    def save(self, name):
        """Save model weights to file"""
        # Ensure the filename ends with .weights.h5
        if not name.endswith('.weights.h5'):
            if name.endswith('.h5'):
                name = name.replace('.h5', '.weights.h5')
            else:
                name = name + '.weights.h5'
        self.model.save_weights(name)