"""

    Not fit for any purpose, use at your own risk.

    Copyright (c) Rex Sutton 2004-2017.

    Reference implementation of Deep Q-Network,

        for discrete control using deep reinforcement learning.

"""
import types
import numpy as np
import tensorflow as tf

import rllab as rl

__default_params_dict__ = {
    'max_episodes': 1,
    'max_episode_steps': None,
    'min_replays': 64,
    'batch_size': 64,
    'eps_max': 1.0,
    'eps_min': 0.1,
    'discount_factor': 0.99,
    'training_interval': 4,
    'copy_interval': 10000,
    'hidden_units': [16, 16],
    'l2_scale': None,
    'activation': tf.nn.relu,
    'initializer': tf.contrib.layers.variance_scaling_initializer(),
    'learning_rate': 0.005,
    'momentum': 0.95,
    'replay_memory_size': int(1e6),
    'soft_copy': False,
    'tau': 0.001,
    'eval_interval': None,
    'double_q': False
}


def _make_q_network(states, num_actions, params, scope_name):
    """
    Build the q network.
    Args:
        states: The batch of states.
        num_actions: The number of actions.
        params: The hyper-parameters.
        scope_name: Variables will be created in this scope.

    Returns: The q values for each state and action,
        a dictionary of the network variables, and the regularization loss.

    """
    kernel_regularizer = tf.contrib.layers.l2_regularizer(params.l2_scale) if params.l2_scale else None
    with tf.variable_scope(scope_name) as scope:
        hidden0 = tf.layers.dense(states, params.hidden_units[0],
                                  activation=params.activation,
                                  kernel_initializer=params.initializer,
                                  kernel_regularizer=kernel_regularizer)
        hidden = tf.layers.dense(hidden0, params.hidden_units[1],
                                 activation=params.activation,
                                 kernel_initializer=params.initializer,
                                 kernel_regularizer=kernel_regularizer)
        outputs = tf.layers.dense(hidden, num_actions,
                                  kernel_initializer=params.initializer,
                                  kernel_regularizer=kernel_regularizer)

    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    # get the regularization loss
    reg_loss = tf.losses.get_regularization_loss(scope_name)
    return outputs, trainable_vars_by_name, reg_loss


def _make_soft_copy_op(tau, target, online):
    """
    Make `soft copy' from online to target.

    Args:
        tau: Weight with which online variable is assigned to target.
        target: The target variable.
        online: The online variable.

    Returns: The copy operation.

    """
    return target.assign_sub(tau * (target - online))


def _make_soft_copy_ops(tau, target_vars, online_vars):
    """
    Make `soft copy' from online to target.

    Args:
        tau: Weight with which online variable is assigned to target.
        target_vars: The target variables.
        online_vars: The online variables.

    Returns: A list of copy operations.

    """
    return [_make_soft_copy_op(tau, target_vars[var_name], online_vars[var_name])
            for var_name in target_vars.keys()]


class _Graph:
    """ The graph of calculations.
    """

    def __init__(self, state_dim, num_actions, params):
        """
        Initialization.

        Args:
            state_dim: The state dimension.
            num_actions: The number of actions.
            params: The hyper-parameters.
        """
        graph = tf.Graph()
        with graph.as_default():
            states = tf.placeholder(tf.float32, [None, state_dim])
            actions = tf.placeholder(tf.int32, shape=[None])
            td_target = tf.placeholder(tf.float32, shape=[None, 1])
            online_q_values, online_vars, reg_loss = _make_q_network(states, num_actions, params,
                                                                     scope_name="online")
            target_q_values, target_vars, _ = _make_q_network(states, num_actions, params,
                                                              scope_name="target")

            soft_copy_op = tf.group(*_make_soft_copy_ops(params.tau, target_vars, online_vars))
            hard_copy_op = tf.group(*[target_var.assign(online_vars[var_name])
                                      for var_name, target_var in target_vars.items()])

            if params.soft_copy:
                copy_online_to_target = soft_copy_op
            else:
                copy_online_to_target = hard_copy_op

            q_value = tf.reduce_sum(online_q_values * tf.one_hot(actions, num_actions),
                                    axis=1, keepdims=True)

            loss = tf.losses.huber_loss(td_target, q_value)
            if params.l2_scale is not None:
                loss += reg_loss
            optimizer = tf.train.MomentumOptimizer(params.learning_rate, params.momentum, use_nesterov=True)
            training_op = optimizer.minimize(loss)

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        self.flow_graph = graph
        self.init_op = init
        self.states = states
        self.actions = actions
        self.td_target = td_target
        self.online_q_values = online_q_values
        self.target_q_values = target_q_values
        self.hard_copy_op = hard_copy_op
        self.copy_online_to_target = copy_online_to_target
        self.training_op = training_op
        self.loss = loss
        self.saver = saver


def make_params(params_dict=None):
    """
    Make hyper-parameters.

    Args:
        params_dict: The parameters, if None then default values will be used.

    Returns:
        The hyper-parameters.

    """
    if params_dict is None:
        params_dict = dict(__default_params_dict__)
    else:
        params_dict = dict(params_dict)

    return types.SimpleNamespace(**params_dict)


class _Agent:
    """ The agent.
    """
    def __init__(self, state_dim, num_actions, params, verbose):
        """
        Initialization.

        Args:
            state_dim: The state dimension.
            num_actions: The number of actions.
            params: The hyper-parameters.
            verbose: If True print convergence information.
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.params = params
        self.verbose = verbose

        self.graph = _Graph(state_dim, num_actions, params=params)
        self.memory = rl.tools.ReplayMemory(size=params.replay_memory_size)
        self.tracker = rl.tools.RewardTracker(params.discount_factor)
        self.sess = tf.Session(graph=self.graph.flow_graph)

        # initialize graph
        self.sess.run(self.graph.init_op)
        # make networks identical
        self.sess.run(self.graph.hard_copy_op)

        self.episode = 0
        self.episode_step = 0
        self.iteration = 0

        self.loss_val = np.nan
        self.total_max_q = np.nan

    def continues(self):
        """ Returns: True if the agent continues to explore.
        """
        return self.episode < self.params.max_episodes

    def reset(self):
        """
        Reset to the beginning of an episode.
        """
        self.tracker.reset()
        self.total_max_q = 0.0
        self.episode_step = 0
        self.episode += 1

    def explore(self, state):
        """
        Args:
            state: The state of the game.

        Returns: The controls.

        """
        q_values = self.sess.run(self.graph.online_q_values, feed_dict={self.graph.states: [state]})
        epsilon = max(self.params.eps_min,
                      self.params.eps_max - (
                              self.params.eps_max - self.params.eps_min) * self.episode / self.params.max_episodes)
        if np.random.rand() < epsilon:
            action = np.random.randint(self.num_actions)  # random action
        else:
            action = np.argmax(q_values)  # optimal action

        self.total_max_q += q_values.max()
        return action

    def step(self, state, action, reward, next_state, done):
        """
        Process a single step.

        Args:
            state: The state.
            action: The action.
            reward: The reward generated.
            next_state: The next state.
            done: True if the episode has terminated.

        Returns: Done, allows the agent to terminate an episode early.

        """
        self.episode_step += 1
        # if early episode termination
        if self.params.max_episode_steps and self.episode_step >= self.params.max_episode_steps:
            done = True
        # track progress
        self.tracker.step(reward)
        # memorize
        self.memory.append((state, action, reward, next_state, 1.0 - done))
        return done

    def training_step(self):
        """
        Run one training step.
        """
        self.iteration += 1
        # if not enough replay memories
        if self.iteration < self.params.min_replays:
            # skip training
            return
        # sample memories
        states_val, action_val, rewards, next_state_val, continues \
            = (rl.tools.sample_memories(self.memory, self.params.batch_size))
        # evaluate the target q
        target_q = self.sess.run(self.graph.target_q_values, feed_dict={self.graph.states: next_state_val})
        # if using double q
        if self.params.double_q:
            online_q = self.sess.run(self.graph.online_q_values, feed_dict={self.graph.states: next_state_val})
            actions = np.argmax(online_q, axis=1)
            max_next_q_values = target_q[np.arange(actions.shape[0]), actions].reshape(-1, 1)
        else:
            max_next_q_values = np.max(target_q, axis=1, keepdims=True)
        # train the online DQN
        td_target = rewards + continues * self.params.discount_factor * max_next_q_values
        _, self.loss_val = self.sess.run([self.graph.training_op, self.graph.loss],
                                         feed_dict={self.graph.states: states_val, self.graph.actions: action_val,
                                                    self.graph.td_target: td_target})
        # copy to target
        if self.params.copy_interval is None or (
                self.params.copy_interval and (self.iteration % self.params.copy_interval == 0)):
            self.sess.run(self.graph.copy_online_to_target)

    def save(self, save_path):
        """
        Save the policy to disk at the path.
        Args:
            save_path: The path.

        """
        self.graph.saver.save(self.sess, save_path)

    def print_progress(self):
        """
        Print progress to screen.
        """
        print(
            '\rE {} S {} TR {:6.2f} G {:6.2f} Loss {:6.5f} AvgQ {:6.2f}'
            ' MinR {:6.2f} MaxR {:6.2f}'.format(
                self.episode, self.episode_step,
                self.tracker.total_reward, self.tracker.discounted_rewards,
                self.loss_val, self.total_max_q / self.episode_step,
                self.tracker.min_reward, self.tracker.max_reward,
                end=""))

    def show_progress(self, game):
        """
        Print progress to screen, by evaluating one episode.
        Args:
            game: The game to evaluate.
        """
        if self.verbose:
            if self.params.eval_interval is not None and (self.episode % self.params.eval_interval == 0):
                self.print_progress()
                # evaluate one run
                state = game.reset()
                self.tracker.reset()
                done = False
                while not done:
                    q_values = self.sess.run(self.graph.target_q_values, feed_dict={self.graph.states: [state]})
                    action = np.argmax(q_values)
                    state, reward, done, _ = game.step(action)
                    self.tracker.step(reward)
                print('****** ', self.tracker.total_reward, self.tracker.discounted_rewards, ' ******')


def solve(game,
          state_dim, num_actions,
          params,
          save_path,
          seed=None,
          verbose=True):
    """
    Use deep reinforcement learning to search for an optimal policy.
    Args:
        game: The game.
        state_dim: The state dimension.
        num_actions: The number of actions.
        params: The hyper-parameters.
        save_path: The path to save the solution to.
        seed: The seed, if not None, reseed tensor flow and numpy.
        verbose: If True print debugging information.
    """
    if seed is not None:
        tf.set_random_seed(seed=seed)
        np.random.seed(seed=seed)
    agent = _Agent(state_dim, num_actions, params, verbose)
    state = game.reset()
    agent.reset()
    while agent.continues():
        action = agent.explore(state)
        next_state, reward, done, info = game.step(action)
        done = agent.step(state, action, reward, next_state, done)
        state = next_state
        if done:
            agent.show_progress(game)
            state = game.reset()
            agent.reset()
        agent.training_step()
    agent.save(save_path)


class RandomControl:
    """
    Random controls.
    """

    def __init__(self, num_actions):
        """
        Initialization.

        Args:
            num_actions: The number of actions.
        """
        self.num_actions = num_actions

    def __call__(self, _):
        return np.random.randint(self.num_actions)


class _GraphFunctor:
    """
    Base class for functions using the graph.
    """

    def __init__(self, state_dim, num_actions, params, restore_path):
        """
        Initialization.
        Args:
            state_dim: The state dimension.
            num_actions: The number of actions.
            params: The hyper-parameters.
            restore_path: The path to restore variables from.
        """
        self.graph = _Graph(state_dim, num_actions, params=params)
        self.sess = tf.Session(graph=self.graph.flow_graph)
        self.graph.saver.restore(self.sess, restore_path)


class Control(_GraphFunctor):
    """
    Use the policy found by DQN as the controls.
    """

    def q_values(self, state):
        """
        Evaluate the Q value for each action, in the state.
        Args:
            state: The state.

        Returns: The Q-values for each action.

        """
        return self.sess.run(self.graph.target_q_values,
                             feed_dict={self.graph.states: [state]}).reshape(-1)

    def value(self, state):
        """
        Evaluate the value, in the state.
        Args:
            state: The state.

        Returns: The value for the state.

        """
        return np.amax(self.q_values(state))

    def __call__(self, state):
        """
        Return the action.
        Args:
            state: The environment state.

        Returns: The action.

        """
        return np.argmax(self.q_values(state))
