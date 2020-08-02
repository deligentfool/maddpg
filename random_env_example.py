from make_env import make_env
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete

# * It is only noted that the form of submit actions by this file. The action for this env is not one-hot form.
env = make_env('simple_reference')

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        agent_actions = []
        for i, agent in enumerate(env.world.agents):
            agent_action_space = env.action_space[i]

            action = agent_action_space.sample()

            if isinstance(agent_action_space, Discrete):
                action_vev = np.zeros(agent_action_space.n)
                action_vec[action] = 1
                agent_actions.append(action_vec)
            else:
                # * for the MultiDiscrete type action element
                action_vev = np.zeros(sum(agent_action_space.high) + env.n)
                start_idx = 0
                for n in range(agent_action_space.shape):
                    action_vev[start_idx + action[n]] = 1
                    start_idx += agent_action_space.high[n]
                agent_actions.append(action_vev)

        observation, reward, done, info = env.step(agent_actions)
        print (observation)
        print (reward)
        print (done)
        print (info)
        print()