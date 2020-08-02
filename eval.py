from model import maddpg
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--scenario",
        help="set the scenario.",
        type=str
    )
    args = parser.parse_args()
    env_id = args.scenario
    os.makedirs('./models/{}'.format(env_id), exist_ok=True)
    test = maddpg(
        env_id=env_id,
        episode=10000,
        learning_rate=1e-3,
        gamma=0.97,
        capacity=10000,
        batch_size=128,
        value_iter=1,
        policy_iter=1,
        rho=0.99,
        render=True,
        episode_len=45,
        train_freq=5,
        entropy_weight=0.0001,
        model_path=False
    )

    test.eval()