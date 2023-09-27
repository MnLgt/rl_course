import gymnasium as gym
from gymnasium import spaces
import json
import numpy as np
from stable_baselines3.common.env_checker import check_env


# Complete class definition with NaN check and removal only in 'temp'
class TempPredictionEnv(gym.Env):
    def __init__(self, file_path):
        super(TempPredictionEnv, self).__init__()

        # Define action and observation space
        # They must be gymnasium.spaces objects
        # Action space is any floating point number
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))

        # observation is year, day and temperature
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))

        # Load the dataset
        with open(file_path, "r") as f:
            dataset = [json.loads(line) for line in f]

        # Create a new dataset without 'temp' NaN values
        self.dataset = [obs for obs in dataset if obs["temp"]]

        print(
            f"Removed {len(dataset) - len(self.dataset)} NaN values from the dataset."
        )

        self.current_step = 0

    def normalize_action(self, action):
        # Denormalize the actions from [-1, 1] to [-50, 50]
        return action * 50

    def denormalize_action(self, action):
        # Normalize the actions from [-50, 50] to [-1, 1]
        return action / 50

    def step(self, action):
        # Execute one time step within the environment
        terminated = False
        truncated = False

        # Denormalize the action
        action = self.normalize_action(action)

        # Get the current observation
        current_obs = self.dataset[self.current_step]

        # Calculate the reward as negative absolute difference between
        # predicted temperature and the actual temperature
        reward = -abs(current_obs["temp"] - action[0])

        self.current_step += 1

        # If we have stepped through all the data, we are done
        if self.current_step >= len(self.dataset):
            terminated = True
            self.current_step = 0

        next_obs = self.dataset[self.current_step]
        next_obs = np.array(
            [next_obs["year"], next_obs["day"], next_obs["temp"]], dtype=np.float32
        )
        return next_obs, reward, terminated, truncated, {}

    def reset(self, seed=None):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        obs = self.dataset[self.current_step]
        return np.array([obs["year"], obs["day"], obs["temp"]], dtype=np.float32), {}


if __name__ == "__main__":
    # Create the environment
    import os

    fp = os.path.join(
        os.path.dirname(__file__), "..", "datasets", "atlantic_sea_temp.jsonl"
    )
    env = TempPredictionEnv(fp)

    # Check the environment
    check_env(env)

    # Perform a random action
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _ = env.step(action)

    # Print the results
    print(f"obs: {obs}")
    print(f"next_obs: {next_obs}")
    print(f"reward: {reward}")
    print(f"terminated: {terminated}")
    print(f"truncated: {truncated}")
