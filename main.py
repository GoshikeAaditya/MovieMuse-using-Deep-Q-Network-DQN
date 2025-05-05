import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Movie Recommendation System using DQN')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--recommend', action='store_true', help='Generate recommendations')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--accuracy', action='store_true', help='Calculate accuracy metric')
    parser.add_argument('--web', action='store_true', help='Start web interface')
    parser.add_argument('--user_id', type=int, help='User ID for recommendations')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--top_k', type=int, default=10, help='Number of recommendations')
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting training...")
        os.system(f"python train.py --episodes {args.episodes}")
    
    if args.evaluate or args.accuracy:
        print("Evaluating model...")
        os.system(f"python recommend.py --evaluate --top_k {args.top_k}")
    
    if args.recommend:
        if args.user_id is None:
            print("Error: Please provide a user_id for recommendations")
            return
        
        print(f"Generating recommendations for user {args.user_id}...")
        os.system(f"python recommend.py --user_id {args.user_id} --top_k {args.top_k}")
    
    if args.web:
        print("Starting web interface...")
        os.system("python web_interface.py")

if __name__ == "__main__":
    main()