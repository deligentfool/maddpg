#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 08:47:40 2018

@author: xuzhiwei
"""

import make_env
from Replay_buffer import Replay_buffer
from Maddpg import Maddpg
import tensorflow as tf
import numpy as np

START_EPISODE = 50000
MAX_EPISODE = 10000000
CAPACITY = 1000000
BATCH_SIZE = 1024
NOISE_BOTTOM = 3

if __name__ == '__main__':
    
    env = make_env.make_env('simple_tag')
    n_feature = 16
    n_action = 2
    n_agent = env.n - 1
    noise_rate = 3
    
    sess = tf.Session()

    agent_1 = Maddpg('agent_1_',n_action,n_feature,n_agent,sess)
    agent_2 = Maddpg('agent_2_',n_action,n_feature,n_agent,sess)
    agent_3 = Maddpg('agent_3_',n_action,n_feature,n_agent,sess)
        
    saver = tf.train.Saver()
    
    buf_1 = Replay_buffer(n_agent,n_action,n_feature,CAPACITY,BATCH_SIZE)
    buf_2 = Replay_buffer(n_agent,n_action,n_feature,CAPACITY,BATCH_SIZE)
    buf_3 = Replay_buffer(n_agent,n_action,n_feature,CAPACITY,BATCH_SIZE)
    
    sess.run(tf.global_variables_initializer())

    reward_info = np.zeros(1000)
    running_reward = np.sum(reward_info)
    
    observation = env.reset()
    
    for episode in range(MAX_EPISODE):        
        action_1 = agent_1.choose_action(np.array(observation[0])[None,:]) + np.random.randn(2) * noise_rate
        action_2 = agent_2.choose_action(np.array(observation[1])[None,:]) + np.random.randn(2) * noise_rate
        action_3 = agent_3.choose_action(np.array(observation[2])[None,:]) + np.random.randn(2) * noise_rate
        
        action = [[0,i[0][0],0,i[0][1],0] for i in [action_1,action_2,action_3]]
        action.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            
        observation_,reward,done,info = env.step(action)
        reward = [i / 5 for i in reward]
        
        reward_info[episode % 1000] = reward[0] + reward[1] + reward[2]
        
        buf_1.add(np.hstack([observation[0],observation[1],observation[2]]),np.hstack([action_1,action_2,action_3]).ravel(),np.hstack([reward[0]]),np.hstack([observation_[0],observation_[1],observation_[2]]))
        buf_2.add(np.hstack([observation[1],observation[2],observation[0]]),np.hstack([action_2,action_3,action_1]).ravel(),np.hstack([reward[1]]),np.hstack([observation_[1],observation_[2],observation_[0]]))
        buf_3.add(np.hstack([observation[2],observation[0],observation[1]]),np.hstack([action_3,action_1,action_2]).ravel(),np.hstack([reward[2]]),np.hstack([observation_[2],observation_[0],observation_[1]]))

        running_reward = 0.9999 * running_reward + 0.0001 * np.sum(reward_info)
        
        if (episode >= START_EPISODE-1):
            sample_index_1 = buf_1.get_sample_index()
            sample_index_2 = buf_2.get_sample_index()
            sample_index_3 = buf_3.get_sample_index()
            
            a_1_2_ = agent_2.choose_next_action(buf_2.get_self_next_observation(sample_index_1))
            a_1_3_ = agent_3.choose_next_action(buf_3.get_self_next_observation(sample_index_1))
            
            a_2_3_ = agent_3.choose_next_action(buf_3.get_self_next_observation(sample_index_2))
            a_2_1_ = agent_1.choose_next_action(buf_1.get_self_next_observation(sample_index_2))
            
            a_3_1_ = agent_1.choose_next_action(buf_1.get_self_next_observation(sample_index_3))
            a_3_2_ = agent_2.choose_next_action(buf_2.get_self_next_observation(sample_index_3))
            
            agent_1.learn(s=buf_1.get_self_observation(sample_index_1),
                          s_=buf_1.get_self_next_observation(sample_index_1),
                          a=buf_1.get_self_action(sample_index_1),
                          r=buf_1.get_reward(sample_index_1),
                          other_action=buf_1.get_other_action(sample_index_1),
                          other_action_=np.hstack([a_1_2_,a_1_3_]),
                          other_S=buf_1.get_other_next_observation(sample_index_1),
                          other_S_=buf_1.get_other_next_observation(sample_index_1))
            
            agent_2.learn(s=buf_2.get_self_observation(sample_index_2),
                          s_=buf_2.get_self_next_observation(sample_index_2),
                          a=buf_2.get_self_action(sample_index_2),
                          r=buf_2.get_reward(sample_index_2),
                          other_action=buf_2.get_other_action(sample_index_2),
                          other_action_=np.hstack([a_2_3_,a_2_1_]),
                          other_S=buf_2.get_other_next_observation(sample_index_2),
                          other_S_=buf_2.get_other_next_observation(sample_index_2))
            
            agent_3.learn(s=buf_3.get_self_observation(sample_index_3),
                          s_=buf_3.get_self_next_observation(sample_index_3),
                          a=buf_3.get_self_action(sample_index_3),
                          r=buf_3.get_reward(sample_index_3),
                          other_action=buf_3.get_other_action(sample_index_3),
                          other_action_=np.hstack([a_3_1_,a_3_2_]),
                          other_S=buf_3.get_other_next_observation(sample_index_3),
                          other_S_=buf_3.get_other_next_observation(sample_index_3))
            
            if noise_rate > NOISE_BOTTOM:
                noise_rate *= 0.9999
            
            if (episode + 1) % 1000 == 0:
                saver.save(sess, './simple_tag_ma_weight/' + str(episode+1) + '.cptk')
                print('episode:{} running_reward:{:.3f} explore:{:.3f}'.format(episode+1,running_reward,noise_rate))
                
            observation = observation_
       
        if (episode + 1) % 250:
            observation = env.reset()