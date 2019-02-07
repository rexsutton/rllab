import argparse
import gym
import numpy as np

import env
import rllab as rl


__state_dim__ = 2
__num_actions__ = 3
__policy_path__ = './mount_car_discrete.ckpt'
__gym_env_name__ = 'MountainCar-v0'

__params__ = rl.dqn.make_params()
__params__.max_episodes = 16384
__params__.eval_interval = 256
__params__.hidden_units = [32, 32]
__params__.min_replays = 4096
__params__.copy_interval = 256
__params__.double_q = True


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

    rl.dqn.solve(game, state_dim=__state_dim__, num_actions=__num_actions__,
                 params=__params__, save_path=__policy_path__, seed=42)


def test():
    gym_env = gym.make(__gym_env_name__)

    game = rl.tools.GameAdaptor(gym_env,
                                transform_observation_func=transform_observation)

    control = rl.dqn.Control(__state_dim__, __num_actions__,
                             params=__params__,
                             restore_path=__policy_path__)

    rl.tools.render(game, control_func=control,
                    path='./' + __gym_env_name__ + '.gif')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--command', help='The command to invoke.',
                        choices=['train', 'test'], default='test')
    args = parser.parse_args()
    cmd_dict = {'train': train, 'test': test}
    _ = cmd_dict[args.command]() if args.command in cmd_dict else parser.print_help()


if __name__ == '__main__':
    main()
