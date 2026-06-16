"""
Environment wrappers and preprocessing utilities for Super Mario Bros agent.
"""

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


def make_mario_env():
    """
    Create and preprocess a Super Mario Bros environment.
    
    Returns:
        env: Preprocessed vectorized environment with frame stacking
    """
    # 1. Create the base environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    
    # 2. Simplify the controls
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # 3. Grayscale
    env = GrayScaleObservation(env, keep_dim=True)
    
    # 4. Wrap inside the Dummy Environment (for vectorization)
    env = DummyVecEnv([lambda: env])
    
    # 5. Stack the frames (4 frames to capture temporal information)
    env = VecFrameStack(env, 4, channels_order='last')
    
    return env


def get_action_name(action_idx):
    """
    Get the action name from the action index.
    
    Args:
        action_idx (int): Index of the action
    
    Returns:
        str: Name of the action
    """
    return str(SIMPLE_MOVEMENT[action_idx])
