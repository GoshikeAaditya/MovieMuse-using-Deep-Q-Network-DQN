# Movie Recommendation System using Deep Q-Network (DQN)

This project implements a movie recommendation system using reinforcement learning with the Deep Q-Network (DQN) algorithm. The system learns to recommend movies to users based on their rating history and movie features.

## Dataset

The project uses the following dataset files:
- `movies.csv`: Contains movie information (ID, title, genres)
- `ratings.csv`: Contains user ratings for movies

## Project Structure

- `data_preprocessing.py`: Handles data loading and preprocessing
- `rl_environment.py`: Implements the reinforcement learning environment
- `dqn_agent.py`: Implements the DQN agent
- `train.py`: Script for training the model
- `recommend.py`: Script for generating recommendations and evaluating the model
- `main.py`: Main script to run the project

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt