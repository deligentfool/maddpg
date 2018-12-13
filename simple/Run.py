#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 08:47:40 2018

@author: xuzhiwei
"""

import make_env
from Replay_buffer import Replay_buffer
from Ddpg import Ddpg
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    
    env = make_env.make_env('simple')
    n_feature = 4
    n_action = 2
    n_agent = env.n
    noise_rate = 0.02

    agent = Ddpg(n_action,n_feature)
    
    saver = tf.train.Saver()
    
    buf = Replay_buffer(n_agent,n_action,n_feature,500000,1024)
    
    agent.sess.run(tf.global_variables_initializer())

    
    reward_info = np.zeros(1000)
    observation = env.reset()

    running_reward = np.sum(reward_info)
    for episode in range(10000000):        
        action_true = agent.choose_action(np.array(observation[0])[None,:]) + np.random.randn(2) * noise_rate
        
        action = [[0,i[0][0],0,i[0][1],0] for i in [action_true]]
            
        observation_,reward,done,info = env.step(action)
           
        reward_info[episode % 1000] = reward[0]
        buf.add(np.array(observation[0]),np.hstack([action_true]).ravel(),np.hstack([reward[0]]),np.array(observation_[0]))
        
        if (episode >= 50000):
            sample_index = buf.get_sample_index()
            
            agent.learn(buf.get_observation(sample_index),buf.get_next_observation(sample_index),
                        buf.get_all_action(sample_index),buf.get_reward(sample_index))
                
            running_reward = 0.9999 * running_reward + 0.0001 * np.sum(reward_info)

            if (episode + 1) % 1000 == 0:
                saver.save(agent.sess, './simple_ma_weight/' + str(episode+1) + '.cptk')
                print('episode:{} reward:{} running_reward:{:.3f}'.format(episode+1,np.sum(reward_info),running_reward))

            
        observation = observation_
        if (episode + 1) % 2000:
            observation = env.reset()