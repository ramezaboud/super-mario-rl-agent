"""
Play Super Mario Bros using a trained RL agent.
"""

import os
import glob
from wrappers import make_mario_env
from agent import MarioAgent


def get_latest_model(train_dir='./train/'):
    """
    Find the latest trained model in the specified directory.
    
    Args:
        train_dir (str): Directory containing model checkponts.
        
    Returns:
        str: Path to the most recently modified model without extension, or None if not found.
    """
    if not os.path.exists(train_dir):
        return None
        
    # Look for all .zip files in the directory
    models = glob.glob(os.path.join(train_dir, '*.zip'))
    
    if not models:
        return None
        
    # Get the latest modified file
    latest_model = max(models, key=os.path.getctime)
    
    # Remove the .zip extension as stable-baselines3 appends it during load
    return latest_model[:-4]


def play_game(model_path=None, episodes=1, max_steps=None):
    """
    Play Super Mario Bros using a trained agent.
    
    Args:
        model_path (str, optional): Path to the trained model. If None, picks the latest from ./train/.
        episodes (int): Number of episodes to play
        max_steps (int): Maximum steps per episode (None for infinite)
    """
    # If no path provided, try to find the latest model
    if model_path is None:
        model_path = get_latest_model('./train/')
        
    # Check if model exists
    if model_path is None or not os.path.exists(model_path + '.zip'):
        print(f"Model not found at {model_path if model_path else './train/'}")
        print("Please train the model first using train.py")
        return
    
    # Create environment
    print("Creating environment...")
    env = make_mario_env()
    
    # Load the agent
    print(f"Loading model from {model_path}...")
    agent = MarioAgent.load(model_path, env)
    
    print(f"Playing {episodes} episode(s)...")
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done:
            # Predict action
            action, _ = agent.predict(state, deterministic=True)
            
            # Take step
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Render the game
            try:
                env.venv.envs[0].env.unwrapped.render()
            except Exception as e:
                # Silently fail on render issues to continue the game
                pass
            
            # Check max steps
            if max_steps is not None and step >= max_steps:
                done = True
            
            # Print progress periodically
            if step % 100 == 0:
                print(f"Step: {step}, Reward: {episode_reward}")
        
        print(f"Episode {episode + 1} finished!")
        print(f"Total steps: {step}, Total reward: {episode_reward}")
    
    env.close()
    print("\nGame finished!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Super Mario Bros with a trained RL Agent')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the trained model (without .zip). If not specified, uses the latest from ./train/')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes to play (default: 1)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum steps per episode (default: unlimited)')
    
    args = parser.parse_args()
    play_game(
        model_path=args.model,
        episodes=args.episodes,
        max_steps=args.max_steps
    )
