### [👉 Stable 3 Baseline](https://stable-baselines3.readthedocs.io/en/master/)
    - SB3 is a set of reliable implementations of reinforcement learning algorithms in PyTorch.

### [👉 RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
    - a training framework for Reinforcement Learning (RL), using Stable Baselines3.

### [👉 Gymnasium](https://gymnasium.farama.org/)
    - The Gymnasium library provides two things:

        - An interface that allows you to create RL environments.
        - A collection of environments (gym-control, atari, box2D...)

### [🤗 RL Leaderboard](https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard)

### [🤗 RL Course Completion](https://huggingface.co/spaces/ThomasSimonini/Check-my-progress-Deep-RL-Course)


### Custom Environments

Environment must have three methods

***reset()***
- called at the beginning of an episode, it returns an observation and a dictionary with additional info (defaults to an empty dict)

***step(action)***
- called to take an action with the environment, 
it returns the next observation, the immediate reward, whether new state is a terminal 
state (episode is finished), whether the max number of timesteps is reached (episode 
is artificially finished), and additional information

***render()***
    - (Optional) 
- which allow to visualize the agent in action.