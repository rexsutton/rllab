"""

    Not fit for any purpose, use at your own risk.

    Copyright (c) Rex Sutton 2004-2019.

    Miscellaneous functions and classes.

"""

import imageio
import numpy as np
import tensorflow as tf


def identity(x):
    """

    Args:
        x:

    Returns:
        x
    """
    return x


class ReplayMemory:
    """ A ring buffer.
    """
    def __init__(self, size, keep=0):
        """
        Initialization.

        Args:
            size: The maximum number of memories to remember.
            keep: This number of initial memories are never overwritten.
        """

        if keep >= size:
            raise ValueError('The number to keep must be less than the maximum length of the replay memory.')

        self.size = size
        self.keep = keep
        self.buf = np.empty(shape=size, dtype=np.object)
        self.index = 0
        self.length = 0
        self.kept = 0

    def append(self, data):
        """
        Add data to the memory.

        Args:
            data: The data to store.
        """
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.size)
        self.kept = min(self.kept + 1, self.keep)
        self.index = max(self.kept, (self.index + 1) % self.size)

    def sample(self, batch_size):
        """
        Sample memories (with replacement).

        Args:
            batch_size:

        Returns: The samples from the memory.

        """
        indices = np.random.randint(self.length, size=batch_size)  # faster
        return self.buf[indices]


class RewardTracker:
    """
        Calculate statistics about the rewards encountered during exploration.

        Maintain the convention that the reward encountered on the first step is not discounted.
    """

    def __init__(self, discount_factor):
        """
        Initialization.

        Args:
            discount_factor: The discount factor (commonly labelled gamma in the literature).
        """
        self.discount_factor = discount_factor

        self.cumulative_discount_factor = None
        self.total_reward = None
        self.discounted_rewards = None
        self.max_reward = None
        self.min_reward = None

    def reset(self):
        """
        Reset to the beginning of an episode.
        """
        self.cumulative_discount_factor = 1.0
        self.total_reward = 0.0
        self.discounted_rewards = 0.0
        self.max_reward = -1e8
        self.min_reward = 1e8

    def step(self, reward):
        """
        Process a single step.

        Args:
            reward: The reward.
        """
        self.total_reward += reward
        self.discounted_rewards += self.cumulative_discount_factor * reward
        self.max_reward = max(self.max_reward, reward)
        self.min_reward = min(self.min_reward, reward)
        self.cumulative_discount_factor *= self.discount_factor


def sample_memories(replay_memory, batch_size):
    """
    Reshape memories into batch-size.

    Args:
        replay_memory: The replay memory.
        batch_size: The number to sample.

    Returns: Stored memories re-shaped to batch sizes.

    """
    cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
    for memory in replay_memory.sample(batch_size):
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


class GameAdaptor:
    """
    Wrapper for transforming inputs and outputs from the reinforcement learning agent.
    """

    def __init__(self,
                 game,
                 transform_control_func=identity,
                 transform_observation_func=identity,
                 transform_reward_func=identity):
        """

        Args:
            game: The game (environment) being wrapped.
            transform_control_func: Function to transform controls output from the agent.
            transform_observation_func: Function to transform observations output from the game.
            transform_reward_func: Function to transform rewards output from game.
        """
        self.transform_controls_func = transform_control_func
        self.transform_observation_func = transform_observation_func
        self.transform_reward_func = transform_reward_func
        self.game = game
        self.obs = None
        self.raw_reward = None

    def reset(self):
        """
        Reset to the beginning of an episode.

        Returns: The state.
        """
        self.obs = self.game.reset()
        state = self.transform_observation_func(self.obs)
        self.raw_reward = None
        return state

    def step(self, controls):
        """
        Process a single step.

        Args:
            controls:

        Returns: The state, reward, done flag and any extra information.

        """
        self.obs, self.raw_reward, done, info = self.game.step(self.transform_controls_func(controls))
        state = self.transform_observation_func(self.obs)
        reward = self.transform_reward_func(self.raw_reward)
        return state, reward, done, info

    def render(self, mode):
        """
        Render the game.
        Args:
            mode: The mode.

        Returns:
            The results of rendering the game.

        """
        return self.game.render(mode)


class AugmentStateWithTau:
    """
    Wrapper that adds time remaining to the end of the episode for games with finite number of steps.

    Adding time dependence removes the effects of state aliasing and improves convergence.

    """
    def __init__(self,
                 game,
                 max_steps,
                 max_time=1.0):
        """

        Args:
            game: The game to wrap.
            max_steps: The maximum number of steps.
            max_time: The units time is measured in, defaults to one episode.
        """
        self.game = game
        self.max_steps = float(max_steps)
        self.max_time = max_time
        self.step_idx = 0
        self.tau = np.empty([1])

    def reset(self):
        """
        Reset to the beginning of an episode.

        Returns: The state.
        """
        self.step_idx = 0
        self.tau[0] = self.max_time
        obs = self.game.reset()
        return np.concatenate((obs, self.tau))

    def step(self, controls):
        """
        Process a single step.

        Args:
            controls:

        Returns: The state, reward, done flag and any extra information.

        """
        self.step_idx += 1
        self.tau[0] = self.max_time * (1.0 - self.step_idx / self.max_steps)
        obs, reward, done, info = self.game.step(controls)
        return np.concatenate((obs, self.tau)), reward, done, info

    def render(self, mode):
        """
        Render the game.
        Args:
            mode: The mode.

        Returns:
            The results of rendering the game.

        """
        return self.game.render(mode)


class ConstantControl:
    """ A policy where the controls have constant value.
    """

    def __init__(self, controls):
        """
        Initialization.

        Args:
            controls: The values of the controls.
        """
        self.controls = controls

    def __call__(self, _):
        """

        Args:
            _: Ignore all arguments.

        Returns: The constant controls.

        """
        return self.controls


class ControlAdaptor:
    """
    Wrapper for transforming outputs from a control function.
    """

    def __init__(self, control_func, pre_process_controls_func):
        """

        Args:
            control_func: The control function.
            pre_process_controls_func: The transforming function.
        """
        self.control = control_func
        self.pre_process_controls_func = pre_process_controls_func

    def __call__(self, state):
        return self.pre_process_controls_func(self.control(state))


def sample(game, control_func, num_episodes, discount_factor=0.99, seed=None):
    """
    Sample rewards from a number of episodes.

    Args:
        game: The game to sample.
        control_func: The control function.
        num_episodes: The number of episodes.
        discount_factor: The discount factor (commonly labelled gamma in the literature).
        seed: The seed, if not none, reseed tensor flow and numpy.

    Returns: The discounted cumulative rewards, and the number of steps, for each episode.

    """
    if seed is not None:
        tf.set_random_seed(seed=seed)
        np.random.seed(seed=seed)

    # run simulations with the saved model
    cumulative_rewards = []
    steps = []
    # for each simulation
    for i in range(num_episodes):
        done = False
        cumulative_reward = 0.0
        df = discount_factor
        step = 0
        state = game.reset()
        # loop
        while not done:
            controls = control_func(state)
            state, reward, done, _ = game.step(controls)
            cumulative_reward += df * reward
            step += 1
            df *= discount_factor
        cumulative_rewards.append(cumulative_reward)
        steps.append(step)
    return np.array(cumulative_rewards), np.array(steps)


def render(game, control_func, path, duration=0.01, seed=None):
    """
    Render a single episode to a gif file.

    Args:
        game: The game to render.
        control_func: The control function.
        path: The path to save the gif to.
        duration: The frame duration.
        seed: The seed, if not None, reseed tensor flow and numpy.

    Returns: The rendered frames and the un-discounted total reward.

    """
    if seed is not None:
        tf.set_random_seed(seed=seed)
        np.random.seed(seed=seed)

    images = []
    total_reward = 0.0
    done = False
    state = game.reset()
    images.append(game.render(mode='rgb_array'))

    while not done:
        controls = control_func(state)
        state, reward, done, _ = game.step(controls)
        image = game.render(mode='rgb_array')
        images.append(image)
        total_reward += reward

    imageio.mimsave(path, images, duration=duration)
    return images, total_reward
