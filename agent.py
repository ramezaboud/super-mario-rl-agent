"""
Mario RL Agent using stable-baselines3 PPO algorithm.
"""

from stable_baselines3 import PPO


class MarioAgent:
    """
    A wrapper class for the PPO agent used to play Super Mario Bros.
    """
    
    def __init__(self, env, learning_rate=1e-6, n_steps=512, verbose=1, tensorboard_log='./logs/'):
        """
        Initialize the Mario Agent.
        
        Args:
            env: The preprocessed game environment
            learning_rate (float): Learning rate for the PPO algorithm
            n_steps (int): Number of steps to collect before updating
            verbose (int): Verbosity level
            tensorboard_log (str): Path for tensorboard logs
        """
        self.env = env
        self.model = PPO(
            'CnnPolicy',
            env,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            learning_rate=learning_rate,
            n_steps=n_steps
        )
    
    def learn(self, total_timesteps=1000000, callback=None):
        """
        Train the agent.
        
        Args:
            total_timesteps (int): Total number of timesteps to train
            callback: Callback function for saving checkpoints
        """
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def predict(self, state, deterministic=False):
        """
        Predict the next action.
        
        Args:
            state: Current game state
            deterministic (bool): Whether to use deterministic action selection
        
        Returns:
            tuple: (action, logprobs)
        """
        return self.model.predict(state, deterministic=deterministic)
    
    def save(self, path):
        """
        Save the model.
        
        Args:
            path (str): Path to save the model
        """
        self.model.save(path)
    
    @staticmethod
    def load(path, env):
        """
        Load a pretrained model.
        
        Args:
            path (str): Path to the saved model
            env: The game environment
        
        Returns:
            MarioAgent: Loaded agent
        """
        agent = MarioAgent(env)
        agent.model = PPO.load(path)
        return agent
