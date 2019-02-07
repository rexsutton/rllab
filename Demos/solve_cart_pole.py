import argparse
import pickle
import gym
import sklearn.preprocessing as pp

import env
import rllab as rl

__state_dim__ = 4
__num_actions__ = 2
__policy_path__ = './cartpole.ckpt'
__gym_env_name__ = 'CartPole-v0'
__prep_path__ = 'CartPole-v0_prep.pkl'

__params__ = rl.dqn.make_params()
__params__.max_episodes = 4096
__params__.eval_interval = 256
__params__.hidden_units = [32, 32]
__params__.min_replays = 4096
__params__.copy_interval = 256
__params__.double_q = True


def transform_reward(reward):
    return reward / 100.0


class RecordState:

    def __init__(self):
        self.records = []

    def __call__(self, state):
        self.records.append(state)
        return state


class SklearnAdaptor:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, arg):
        return self.transform.transform([arg])[0]


def make_state_prep(gym_env):
    recorder = RecordState()
    game = rl.tools.GameAdaptor(gym_env, transform_observation_func=recorder)
    rl.tools.sample(game, control_func=rl.dqn.RandomControl(num_actions=__num_actions__), num_episodes=2048)
    prep = pp.RobustScaler()
    prep.fit(recorder.records)
    return prep


def train():
    gym_env = gym.make(__gym_env_name__)

    prep = make_state_prep(gym_env)

    with open(__prep_path__, 'wb') as filestream:
        pickle.dump(prep, filestream)

    with open(__prep_path__, 'rb') as filestream:
        prep = pickle.load(filestream)

    game = rl.tools.GameAdaptor(gym_env,
                                transform_observation_func=SklearnAdaptor(prep),
                                transform_reward_func=transform_reward)

    rl.dqn.solve(game, state_dim=__state_dim__, num_actions=__num_actions__,
                 params=__params__, save_path=__policy_path__, seed=42)


def test():
    gym_env = gym.make(__gym_env_name__)

    with open(__prep_path__, 'rb') as filestream:
        prep = pickle.load(filestream)

    game = rl.tools.GameAdaptor(gym_env,
                                transform_observation_func=SklearnAdaptor(prep))

    control = rl.dqn.Control(__state_dim__, __num_actions__,
                             params=__params__,
                             restore_path=__policy_path__)

    rl.tools.render(game, control_func=control, path='./' + __gym_env_name__ + '.gif')


def rand():
    gym_env = gym.make(__gym_env_name__)

    with open(__prep_path__, 'rb') as filestream:
        prep = pickle.load(filestream)

    game = rl.tools.GameAdaptor(gym_env,
                                transform_observation_func=SklearnAdaptor(prep))

    control = rl.dqn.RandomControl(__num_actions__)

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
