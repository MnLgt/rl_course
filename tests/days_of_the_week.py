import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from datetime import datetime
import pandas as pd

# Here we will create a simple environment to train an agent on predicing the day of the week
# given the date. The observation is the date, and the action is the day of the week.
# The reward is 1 if the prediction is correct, 0 otherwise.
# The episode terminates after 10 steps.


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, df):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        # Date will be represented as days since 1900-01-01
        self.df = df
        self.current_date = None

        # Action space: 7 days of the week (0: Monday, 1: Tuesday, ... , 6: Sunday)
        self.action_space = spaces.Discrete(7)

        # State space: Number of days since reference date
        self.observation_space = spaces.Discrete(366)

        self.done = False
        self.step_count = 0
        self.max_steps = 10

        self.MONDAY = 0
        self.TUESDAY = 1
        self.WEDNESDAY = 2
        self.THURSDAY = 3
        self.FRIDAY = 4
        self.SATURDAY = 5
        self.SUNDAY = 6

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed, options=options)
        self.terminated = False
        self.step_count = 0
        self.info = {}

        self.current_date = pd.Timestamp(np.random.choice(self.df.index)).dayofyear - 1
        return self.current_date, {}

    def step(self, action):
        correct_day = self.df.iloc[self.current_date]["weekday"]

        if action == correct_day:
            reward = 1
        else:
            reward = -1

        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False

        # Moving to next state
        next_state = self.reset() if terminated else self.current_date

        return next_state, reward, terminated, truncated, {}

    def render(self, mode="human"):
        # Just print the current date for simplicity
        print(f"Current Date: {self.current_date}")


if __name__ == "__main__":
    # Create a sample dataframe
    date_range = pd.date_range(start="2022-01-01", end="2022-12-31")
    df = pd.DataFrame(date_range, columns=["date"])
    df["weekday"] = df["date"].dt.dayofweek  # 0: Monday, 1: Tuesday, ..., 6: Sunday
    df.set_index("date", inplace=True)

    env = CustomEnv(df)
    check_env(env, warn=True)
