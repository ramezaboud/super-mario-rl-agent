"""
Training script for Super Mario Bros RL agent.
"""

import os
from stable_baselines3.common.callbacks import BaseCallback
from wrappers import make_mario_env
from agent import MarioAgent


class TrainAndLoggingCallback(BaseCallback):
    """
    Custom callback for saving model checkpoints during training.
    """
    
    def __init__(self, check_freq, save_path, verbose=1):
        """
        Initialize the callback.
        
        Args:
            check_freq (int): Save model every check_freq steps
            save_path (str): Directory to save checkpoints
            verbose (int): Verbosity level
        """
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        """Initialize the callback by creating save directory."""
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        """Save model at regular intervals."""
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
        return True


def train_agent(total_timesteps=1000000, checkpoint_freq=10000, learning_rate=1e-6, n_steps=512):
    """
    Train the Mario agent.
    
    Args:
        total_timesteps (int): Total timesteps to train
        checkpoint_freq (int): Save checkpoint every N steps
        learning_rate (float): Learning rate for PPO
        n_steps (int): Number of steps per update
    """
    # Create directories
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'
    
    # Create environment
    print("Creating environment...")
    env = make_mario_env()
    
    # Create agent
    print("Initializing agent...")
    agent = MarioAgent(env, learning_rate=learning_rate, n_steps=n_steps, tensorboard_log=LOG_DIR)
    
    # Create callback
    callback = TrainAndLoggingCallback(check_freq=checkpoint_freq, save_path=CHECKPOINT_DIR)
    
    # Train the agent
    print(f"Training agent for {total_timesteps} timesteps...")
    agent.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save the final model
    agent.save(os.path.join(CHECKPOINT_DIR, 'final_model'))
    print(f"Training completed! Models saved to {CHECKPOINT_DIR}")
    
    env.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the Super Mario Bros RL Agent')
    parser.add_argument('--timesteps', type=int, default=1000000,
                        help='Total timesteps to train (default: 1000000)')
    parser.add_argument('--checkpoint-freq', type=int, default=10000,
                        help='Save checkpoint every N steps (default: 10000)')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='Learning rate (default: 1e-6)')
    parser.add_argument('--n-steps', type=int, default=512,
                        help='Number of steps per update (default: 512)')
    
    args = parser.parse_args()
    train_agent(
        total_timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
        learning_rate=args.lr,
        n_steps=args.n_steps
    )
