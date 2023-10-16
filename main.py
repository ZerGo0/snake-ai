import gymnasium as gym
import uuid

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import env_checker

from pathlib import Path

from gymEnv import GymEnv


def make_env(env_id: str, id: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
        env = GymEnv(str(id))
        env.reset(seed=seed + id)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    env_id = "ai-v1"
    num_cpu = 10
    # session_id = str(uuid.uuid4())[:8]
    # session_path = Path(f"session_{session_id}")

    # env = GymEnv("0")
    # env_checker.check_env(env)
    # exit()
    env = SubprocVecEnv([make_env(env_id, id) for id in range(num_cpu)])
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path="./ai_model", name_prefix=env_id
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

    model.learn(
        total_timesteps=200_000_000,
        callback=checkpoint_callback,
        progress_bar=True,
    )
