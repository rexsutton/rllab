"""
    Not fit for any purpose, use at your own risk.

    Copyright (c) Rex Sutton 2004-2017.

    Reference implementation of Deep Deterministic Policy Gradients,

     for continuous control with deep reinforcement learning.

"""

import types
import numpy as np
import tensorflow as tf
import rllab as rl


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


def _make_hard_copy_ops(target_vars, online_vars):
    """
    Make copy from online to target.

    Args:
        target_vars: The target variables.
        online_vars: The online variables.

    Returns: A list of copy operations.

    """
    return [(target_vars[var_name].assign(online_vars[var_name]))
            for var_name in target_vars.keys()]


def _make_optimizer(use_adam, learning_rate):
    """
    Make optimizer according to hyper-parameters.
    Args:
        use_adam: If true use Adam.
        learning_rate: The learning rate.

    Returns: The optimizer.

    """
    if use_adam:
        ret = tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        ret = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                         momentum=0.95,
                                         use_nesterov=True)
    return ret


def _make_actor_network(states, num_controls, params, scope_name):
    """
    Make the actor network.

    Args:
        states: The batch of states.
        num_controls: The number of controls.
        params: The hyper-parameters.
        scope_name: : Variables will be created in this scope.

    Returns:
        The controls for each state in the batch.
    """
    scope_key = scope_name + '/'
    with tf.variable_scope(scope_name):
        hidden_1 = tf.layers.dense(states,
                                   units=params.actor_hidden_units[0],
                                   activation=None,
                                   kernel_initializer=params.initializer)
        if params.use_batch_norm:
            hidden_1 = tf.layers.layer_norm(hidden_1, center=True, scale=True)
        hidden_1 = params.activation(hidden_1)
        hidden_2 = tf.layers.dense(hidden_1,
                                   units=params.actor_hidden_units[1],
                                   activation=None,
                                   kernel_initializer=params.initializer)
        if params.use_batch_norm:
            hidden_2 = tf.layers.layer_norm(hidden_2, center=True, scale=True)
        hidden_2 = params.activation(hidden_2)
        controls = tf.layers.dense(hidden_2,
                                   num_controls,
                                   use_bias=True,
                                   activation=tf.nn.tanh,
                                   kernel_initializer=params.initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_key)
    trainable_vars_by_name = {var.name[len(scope_name):]: var for var in trainable_vars}
    return controls, trainable_vars_by_name


def _make_critic_network(states, controls, params, scope_name):
    """
    Make the critic network.

    Args:
        states: The batch of states.
        controls: The corresponding batch of controls.
        params: The hyper-parameters.
        scope_name: : Variables will be created in this scope.

    Returns:
        The maximum Q for each state in the batch.
    """
    scope_key = scope_name + '/'
    kernel_regularizer = tf.contrib.layers.l2_regularizer(params.critic_l2_scale) if params.critic_l2_scale else None
    with tf.variable_scope(scope_name):
        hidden_1 = tf.layers.dense(states,
                                   units=params.critic_hidden_units[0],
                                   activation=None,
                                   kernel_initializer=params.initializer,
                                   kernel_regularizer=kernel_regularizer)
        if params.use_batch_norm:
            hidden_1 = tf.layers.layer_norm(hidden_1, center=True, scale=True)
        hidden_1 = params.activation(hidden_1)
        in_controls = controls
        if params.hide_controls:
            in_controls = tf.layers.dense(controls,
                                          units=params.critic_hidden_units[0],
                                          activation=None,
                                          kernel_initializer=params.initializer,
                                          kernel_regularizer=kernel_regularizer)
            if params.use_batch_norm:
                in_controls = tf.layers.layer_norm(in_controls, center=True, scale=True)
            in_controls = params.activation(in_controls)
        concat = tf.concat(axis=1, values=[hidden_1, in_controls])
        # another hidden layer
        hidden_2 = tf.layers.dense(concat,
                                   units=params.critic_hidden_units[1],
                                   activation=None,
                                   kernel_initializer=params.initializer,
                                   kernel_regularizer=kernel_regularizer)
        if params.use_batch_norm:
            hidden_2 = tf.layers.layer_norm(hidden_2, center=True, scale=True)
        hidden_2 = params.activation(hidden_2)
        # fully connected layer
        outputs = tf.layers.dense(hidden_2, 1,
                                  activation=None,
                                  kernel_initializer=params.initializer,
                                  kernel_regularizer=kernel_regularizer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_key)
    trainable_vars_by_name = {var.name[len(scope_name):]: var for var in trainable_vars}
    reg_loss = tf.losses.get_regularization_loss(scope_name)
    return outputs, trainable_vars_by_name, reg_loss


__default_params_dict__ = {'max_episodes': 1,
                           'max_episode_steps': None,
                           'batch_size': 64,
                           'min_replays': 64,
                           'noise_theta': 0.15,
                           'noise_sigma': 0.2,
                           'discount_factor': 0.99,
                           'tau': 0.001,
                           'actor_hidden_units': [400, 300],
                           'critic_hidden_units': [400, 300],
                           'activation': tf.nn.relu,
                           'initializer': tf.contrib.layers.variance_scaling_initializer(),
                           'critic_l2_scale': None,
                           'use_adam': False,
                           'actor_learning_rate': 0.01,
                           'critic_learning_rate': 0.005,
                           'use_batch_norm': False,
                           'hide_controls': False,
                           'replay_memory_size': int(1e6),
                           'replay_memory_keep': 0,
                           'eval_interval': None,
                           'num_batches': 1,
                           'make_actor_network_func': _make_actor_network,
                           'make_critic_network_func': _make_critic_network,
                           }


class _Graph:
    """ The graph of calculations.
    """
    def __init__(self, state_dim, num_controls, params):
        """
        Initialization.

        Args:
            state_dim: The state dimension.
            num_controls: The number of controls.
            params: The hyper-parameters.
        """
        graph = tf.Graph()
        with graph.as_default():
            # place holders
            states = tf.placeholder(tf.float32, (None, state_dim))
            td_target = tf.placeholder(tf.float32, shape=[None, 1])
            # make online networks
            actor_outputs, actor_vars = params.make_actor_network_func(
                states, num_controls, params, 'actor')
            critic_outputs, critic_vars, critic_reg_loss = params.make_critic_network_func(
                states, actor_outputs, params, 'critic')
            # make target networks
            target_actor_outputs, target_actor_vars = params.make_actor_network_func(states, num_controls,
                                                                                     params, 'actor_target')
            target_critic_outputs, target_critic_vars, _ = params.make_critic_network_func(
                states, target_actor_outputs, params, 'critic_target')
            # make hard copy op
            hard_copy_op = tf.group(*(_make_hard_copy_ops(target_actor_vars, actor_vars)
                                      + _make_hard_copy_ops(target_critic_vars, critic_vars)))
            # make soft copy operations
            copy_online_to_target = tf.group(*(_make_soft_copy_ops(params.tau, target_actor_vars, actor_vars)
                                               + _make_soft_copy_ops(params.tau, target_critic_vars, critic_vars)))

            critic_loss = tf.losses.huber_loss(td_target, critic_outputs)
            if params.critic_l2_scale is not None:
                critic_loss += critic_reg_loss
            critic_optimizer = _make_optimizer(params.use_adam, params.critic_learning_rate)
            critic_training_op = critic_optimizer.minimize(critic_loss,
                                                           var_list=list(critic_vars.values()))
            # train the actor (equivalent to Silver's derivation but easier)
            neg_mean_q = -tf.reduce_mean(critic_outputs)  # + actor_reg_loss
            actor_optimizer = _make_optimizer(params.use_adam, params.actor_learning_rate)
            actor_training_op = actor_optimizer.minimize(neg_mean_q,
                                                         var_list=list(actor_vars.values()))
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        self.flow_graph = graph
        self.init_op = init
        self.states = states
        self.td_target = td_target
        self.actor_outputs = actor_outputs
        self.target_actor_outputs = target_actor_outputs
        self.target_critic_outputs = target_critic_outputs
        self.copy_online_to_target = copy_online_to_target
        self.hard_copy_op = hard_copy_op
        self.critic_loss = critic_loss
        self.critic_reg_loss = critic_reg_loss
        self.critic_training_op = critic_training_op
        self.neg_mean_q = neg_mean_q
        self.actor_training_op = actor_training_op
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
    def __init__(self, state_dim, num_controls, params, verbose):
        """
        Initialization.

        Args:
            state_dim: The state dimension.
            num_controls: The number of controls.
            params: The hyper-parameters.
            verbose: If True print convergence information.
        """
        self.state_dim = state_dim
        self.num_controls = num_controls
        self.params = params
        self.verbose = verbose

        self.graph = _Graph(state_dim, num_controls, params=params)
        self.memory = rl.tools.ReplayMemory(size=params.replay_memory_size, keep=params.replay_memory_keep)
        self.tracker = rl.tools.RewardTracker(params.discount_factor)
        self.noise = np.zeros([num_controls])
        self.sess = tf.Session(graph=self.graph.flow_graph)

        # initialize graph
        self.sess.run(self.graph.init_op)
        # make networks identical
        self.sess.run(self.graph.hard_copy_op)

        self.episode = 0
        self.episode_step = 0
        self.iteration = 0

        self.reg_loss_val = np.nan
        self.critic_loss_val = np.nan
        self.mean_q_val = np.nan

    def continues(self):
        """ Returns: True if the agent continues to explore.
        """
        return self.episode < self.params.max_episodes

    def reset(self):
        """
        Reset to the beginning of an episode.
        """
        self.tracker.reset()
        self.episode += 1
        self.episode_step = 0

    def explore(self, state):
        """
        Args:
            state: The state of the game.

        Returns: The controls.

        """
        controls = self.sess.run(self.graph.actor_outputs,
                                 feed_dict={self.graph.states: np.array([state])}).reshape(-1)
        # update exploration noise
        self.noise -= ((self.params.noise_theta * self.noise) - self.params.noise_sigma * np.random.randn(
            self.num_controls))
        # add in the exploration noise
        weight = float(self.episode) / float(self.params.max_episodes)
        if weight < 1.0:
            controls = controls * weight + (1.0 - weight) * self.noise
        controls = np.maximum(controls, -1.0)
        controls = np.minimum(controls, 1.0)
        return controls

    def step(self, state, controls, reward, next_state, done):
        """
        Process a single step.

        Args:
            state: The state.
            controls: The controls.
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
        self.memory.append((state, controls, reward, next_state, 1.0 - done))
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
        # for each batch
        for _ in range(self.params.num_batches):
            # sample memories
            mem_states, mem_controls, mem_rewards, mem_next_states, mem_continues = \
                (rl.tools.sample_memories(self.memory, self.params.batch_size))
            # train the critic
            max_q = self.sess.run(self.graph.target_critic_outputs, feed_dict={self.graph.states: mem_next_states})
            td_target = mem_rewards + mem_continues * self.params.discount_factor * max_q
            self.reg_loss_val, self.critic_loss_val, _ = self.sess.run(
                [self.graph.critic_reg_loss, self.graph.critic_loss, self.graph.critic_training_op],
                feed_dict={self.graph.states: mem_states, self.graph.actor_outputs: mem_controls,
                           self.graph.td_target: td_target})
            # train the actor
            neg_mean_q_val, _ = self.sess.run([self.graph.neg_mean_q, self.graph.actor_training_op],
                                              feed_dict={self.graph.states: mem_states})
            self.mean_q_val = -1.0 * neg_mean_q_val
        # copy to target
        self.sess.run(self.graph.copy_online_to_target)

    def save(self, save_path):
        """
        Save the policy to disk at the path.
        Args:
            save_path: The path.

        """
        self.graph.saver.save(self.sess, save_path)

    def _print_progress(self):
        """
        Print progress to screen.
        """
        print(
            'E {} S {} TR {:6.2f} G {:6.2f} Reg {:6.5f} Loss {:6.5f} AvgQ {:6.2f}'
            ' MinR {:6.2f} MaxR {:6.2f}'.format(
                self.episode, self.episode_step, self.tracker.total_reward, self.tracker.discounted_rewards,
                self.reg_loss_val, self.critic_loss_val, self.mean_q_val,
                self.tracker.min_reward, self.tracker.max_reward))

    def show_progress(self, game):
        """
        Print progress to screen, by evaluating one episode.
        Args:
            game: The game to evaluate.
        """
        if self.verbose:
            if self.params.eval_interval is not None and (self.episode % self.params.eval_interval == 0):
                self._print_progress()
                # evaluate one run
                state = game.reset()
                self.tracker.reset()
                done = False
                while not done:
                    controls = self.sess.run(self.graph.target_actor_outputs,
                                             feed_dict={self.graph.states: np.array([state])}).reshape(-1)
                    state, reward, done, _ = game.step(controls)
                    self.tracker.step(reward)
                print('****** ', self.tracker.total_reward, self.tracker.discounted_rewards, ' ******')


def solve(game, state_dim, num_controls, params, save_path, seed=None, verbose=True):
    """
    Use deep reinforcement learning to search for an optimal policy.
    Args:
        game: The game.
        state_dim: The state dimension.
        num_controls: The number of controls.
        params: The hyper-parameters.
        save_path: The path to save the solution to.
        seed: The seed, if not None, reseed tensor flow and numpy.
        verbose: If True print debugging information.
    """
    if seed is not None:
        tf.set_random_seed(seed=seed)
        np.random.seed(seed=seed)
    agent = _Agent(state_dim, num_controls, params, verbose)
    state = game.reset()
    agent.reset()
    while agent.continues():
        controls = agent.explore(state)
        next_state, reward, done, _ = game.step(controls)
        done = agent.step(state, controls, reward, next_state, done)
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
    def __init__(self, num_controls):
        """
        Initialization.

        Args:
            num_controls: The number of controls.
        """
        self.num_controls = num_controls

    def __call__(self, _):
        """

        Args:
            _: Ignore arguments.

        Returns: Random controls.

        """
        return np.random.uniform(-1.0, 1.0, self.num_controls)


class _GraphFunctor:
    """
    Base class for functions using the graph.
    """

    def __init__(self, state_dim, num_controls, params, restore_path):
        """
        Initialization.
        Args:
            state_dim: The state dimension.
            num_controls: The number of controls.
            params: The hyper-parameters.
            restore_path: The path to restore variables from.
        """
        self.graph = _Graph(state_dim, num_controls, params=params)
        self.sess = tf.Session(graph=self.graph.flow_graph)
        self.graph.saver.restore(self.sess, restore_path)


class Control(_GraphFunctor):
    """
    Use the policy found by DDPG as the controls.
    """
    def __call__(self, state):
        """
        Return the controls.
        Args:
            state: The environment state.

        Returns: The control values.

        """
        return self.sess.run(self.graph.target_actor_outputs,
                             feed_dict={self.graph.states: [state]}).reshape(-1)


class Value(_GraphFunctor):
    """
    Return the value (estimated maximum Q) for the state, following the policy found by DDPG.
    """

    def __call__(self, state):
        """
        Return the controls.
        Args:
            state: The environment state.

        Returns: The value.

        """
        return self.sess.run(self.graph.target_critic_outputs,
                             feed_dict={self.graph.states: [state]})[0]
