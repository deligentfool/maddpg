from make_env import make_env
import numpy as np

env = make_env('simple_tag')

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        agent_actions = []
        for i, agent in enumerate(env.world.agents):
            # This is a Discrete
            # https://github.com/openai/gym/blob/master/gym/spaces/discrete.py
            agent_action_space = env.action_space[i]

            # Sample returns an int from 0 to agent_action_space.n
            action = agent_action_space.sample()

            # Environment expects a vector with length == agent_action_space.n
            # containing 0 or 1 for each action, 1 meaning take this action
            action_vec = np.zeros(agent_action_space.n)
            action_vec[action] = 1
            agent_actions.append(action_vec)

        # Each of these is a vector parallel to env.world.agents, as is agent_actions
        observation, reward, done, info = env.step(agent_actions)
        print (observation)
        print (reward)
        print (done)
        print (info)
        print()