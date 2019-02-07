import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np

import env
import rllab as rl

__state_dim__ = 2
__num_controls__ = 1
__policy_path__ = './mountain_car.ckpt'
__gym_env_name__ = 'MountainCarContinuous-v0'

__params__ = rl.ddpg.make_params()
__params__.max_episodes = 1024
__params__.actor_hidden_units = [8, 8]
__params__.critic_hidden_units = [8, 16]
__params__.eval_interval = 64
__params__.min_replays = 4096
__params__.replay_memory_size = 100000
__params__.replay_memory_keep = 16384


def transform_observation(obs):
    ret = np.zeros([2])
    ret[0] = (0.3 + obs[0]) / 0.9
    ret[1] = obs[1] / 0.07
    return ret


def transform_reward(reward):
    return reward / 100.0


def train():
    gym_env = gym.make(__gym_env_name__)

    game = rl.tools.GameAdaptor(gym_env,
                                transform_observation_func=transform_observation,
                                transform_reward_func=transform_reward)

    rl.ddpg.solve(game, state_dim=__state_dim__, num_controls=__num_controls__, params=__params__,
                  save_path=__policy_path__, seed=42)


def plot_controls():
    control_func = rl.ddpg.Control(state_dim=__state_dim__, num_controls=__num_controls__, params=__params__,
                                   restore_path=__policy_path__)

    x_axis = np.linspace(-1.2, 0.6, 100)  # position
    y_axis = np.linspace(-0.07, 0.07, 101)  # velocity
    surface = np.zeros([len(y_axis), len(x_axis)])

    for i, y_val in enumerate(y_axis):
        for j, x_val in enumerate(x_axis):
            controls = control_func(transform_observation(np.array([x_val, y_val])))
            surface[i, j] = controls[0]

    fig = plt.figure()
    subplot = fig.add_subplot(111)
    image = subplot.imshow(surface, cmap='jet', interpolation='nearest', origin='lower', vmin=-1.0, vmax=1.0)
    subplot.set_xlabel("position")
    subplot.set_ylabel("velocity")
    fig.colorbar(image)
    plt.show()


def plot_value():
    value_func = rl.ddpg.Value(state_dim=__state_dim__, num_controls=__num_controls__, params=__params__,
                               restore_path=__policy_path__)

    x_axis = np.linspace(-1.2, 0.6, 100)  # position
    y_axis = np.linspace(-0.07, 0.07, 101)  # velocity
    surface = np.zeros([len(y_axis), len(x_axis)])

    for i, y_val in enumerate(y_axis):
        for j, x_val in enumerate(x_axis):
            surface[i, j] = 100.0 * value_func(transform_observation(np.array([x_val, y_val])))

    fig = plt.figure()
    subplot = fig.add_subplot(111)
    image = subplot.imshow(surface, cmap='jet', interpolation='nearest', origin='lower')
    subplot.set_xlabel("position")
    subplot.set_ylabel("velocity")
    fig.colorbar(image)
    plt.show()


def test():
    gym_env = gym.make(__gym_env_name__)

    plot_value()
    plot_controls()

    game = rl.tools.GameAdaptor(gym_env,
                                transform_observation_func=transform_observation)

    control = rl.ddpg.Control(state_dim=__state_dim__, num_controls=__num_controls__, params=__params__,
                              restore_path=__policy_path__)

    rl.tools.render(game, control_func=control, path='./' + __gym_env_name__ + '.gif')


def rand():
    gym_env = gym.make(__gym_env_name__)

    game = rl.tools.GameAdaptor(gym_env,
                                transform_observation_func=transform_observation)

    control = rl.ddpg.RandomControl(num_controls=__num_controls__)

    rl.tools.render(game, control_func=control, path='./' + __gym_env_name__ + '_random.gif')


def main():
    cmd_dict = {'train': train, 'test': test, 'rand': rand}
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--command', help='The command to invoke.',
                        choices=cmd_dict.keys(), default='test')
    args = parser.parse_args()
    _ = cmd_dict[args.command]() if args.command in cmd_dict else parser.print_help()


if __name__ == '__main__':
    main()
