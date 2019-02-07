import argparse
import gym
import numpy as np

import env
import rllab as rl


__state_dim__=4
__num_controls__=1
__policy_path__ = './time_dependent_pendulum.ckpt'
__gym_env_name__ = 'Pendulum-v0'

__params__ = rl.ddpg.make_params()
__params__.max_episodes = 1024
__params__.actor_hidden_units = [16, 16]
__params__.critic_hidden_units = [16, 32]
__params__.eval_interval = 64
__params__.min_replays = 1024


def transform_observation(obs):
    ret = np.zeros([4])
    ret[0] = obs[0]
    ret[1] = obs[1]
    ret[2] = obs[2] / 8.0
    ret[3] = obs[3]
    return ret


def transform_control(control):
    return 2.0 * control


def transform_reward(reward):
    return reward / 100.0


def train():
    gym_env = gym.make(__gym_env_name__)

    time_dependent_env = rl.tools.AugmentStateWithTau(gym_env, max_steps=200)

    game = rl.tools.GameAdaptor(time_dependent_env,
                                transform_control_func=transform_control,
                                transform_observation_func=transform_observation,
                                transform_reward_func=transform_reward)

    rl.ddpg.solve(game, state_dim=__state_dim__, num_controls=__num_controls__, params=__params__, save_path=__policy_path__, seed=42)


def test():
    gym_env = gym.make(__gym_env_name__)

    time_dependent_env = rl.tools.AugmentStateWithTau(gym_env, max_steps=200)

    game = rl.tools.GameAdaptor(time_dependent_env,
                                transform_control_func=transform_control,
                                transform_observation_func=transform_observation,
                                transform_reward_func=transform_reward)

    control = rl.ddpg.Control(state_dim=__state_dim__, num_controls=__num_controls__, params=__params__, restore_path=__policy_path__)

    rl.tools.render(game, control_func=control, path='./' + 'time_dependent_' + __gym_env_name__ + '.gif')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--command', help='The command to invoke.',
                        choices=['train', 'test'], default='test')
    args = parser.parse_args()
    cmd_dict = {'train': train, 'test': test}
    _ = cmd_dict[args.command]() if args.command in cmd_dict else parser.print_help()


if __name__ == '__main__':
    main()
