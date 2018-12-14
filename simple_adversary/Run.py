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

START_EPISODE = 5000
MAX_EPISODE = 10000000
CAPACITY = 1000000
BATCH_SIZE = 256
NOISE_BOTTOM = 3


if __name__ == '__main__':
    
    env = make_env.make_env('simple_adversary')
    n_ad_feature = 8
    n_ga_feature = 10
    n_action = 2
    n_agent = env.n
    noise_rate = 3
    
    sess = tf.Session()

    advag = Maddpg('advag',sess,n_agent,n_action,n_ad_feature)
    goodag_1 = Maddpg('goodag_1',sess,n_agent,n_action,n_ga_feature)
    goodag_2 = Maddpg('goodag_2',sess,n_agent,n_action,n_ga_feature)
    
    saver = tf.train.Saver()
    
    buf_1 = Replay_buffer(n_agent,n_action,n_ad_feature,CAPACITY,BATCH_SIZE)
    buf_2 = Replay_buffer(n_agent,n_action,n_ga_feature,CAPACITY,BATCH_SIZE)
    buf_3 = Replay_buffer(n_agent,n_action,n_ga_feature,CAPACITY,BATCH_SIZE)
    
    sess.run(tf.global_variables_initializer())

    
    reward_info = np.zeros(1000)
    observation = env.reset()

    running_reward = np.sum(reward_info)
    for episode in range(MAX_EPISODE):        
        action_1 = advag.choose_action(np.array(observation[0])[None,:]) + np.random.randn(2) * noise_rate
        action_2 = goodag_1.choose_action(np.array(observation[1])[None,:]) + np.random.randn(2) * noise_rate
        action_3 = goodag_2.choose_action(np.array(observation[2])[None,:]) + np.random.randn(2) * noise_rate
        
        action = [[0,i[0][0],0,i[0][1],0] for i in [action_1,action_2,action_3]]
            
        observation_,reward,done,info = env.step(action)
        
        reward_info[episode % 1000] = reward[1] + reward[2]
        buf_1.add(np.array(observation[0]),np.hstack([action_1,action_2,action_3]).ravel(),np.hstack([reward[0]]),np.array(observation_[0]))
        buf_2.add(np.array(observation[1]),np.hstack([action_2,action_3,action_1]).ravel(),np.hstack([reward[1]]),np.array(observation_[1]))
        buf_3.add(np.array(observation[2]),np.hstack([action_3,action_1,action_2]).ravel(),np.hstack([reward[2]]),np.array(observation_[2]))
        running_reward = 0.9999 * running_reward + 0.0001 * np.sum(reward_info)
        if (episode >= START_EPISODE-1):
            sample_index_1 = buf_1.get_sample_index()
            sample_index_2 = buf_2.get_sample_index()
            sample_index_3 = buf_3.get_sample_index()
            
            a_1_2_ = goodag_1.choose_next_action(buf_2.get_next_observation(sample_index_1))
            a_1_3_ = goodag_2.choose_next_action(buf_3.get_next_observation(sample_index_1))
            
            a_2_3_ = goodag_2.choose_next_action(buf_3.get_next_observation(sample_index_2))
            a_2_1_ = advag.choose_next_action(buf_1.get_next_observation(sample_index_2))

            a_3_1_ = advag.choose_next_action(buf_1.get_next_observation(sample_index_3))
            a_3_2_ = goodag_1.choose_next_action(buf_2.get_next_observation(sample_index_3))
            

            advag.learn(s=buf_1.get_observation(sample_index_1),
                        s_=buf_1.get_next_observation(sample_index_1),
                        a=buf_1.get_action(sample_index_1),
                        r=buf_1.get_reward(sample_index_1),
                        other_action=buf_1.get_other_action(sample_index_1),
                        other_action_=np.hstack([a_1_2_,a_1_3_]))
                

            goodag_1.learn(s=buf_2.get_observation(sample_index_2),
                        s_=buf_2.get_next_observation(sample_index_2),
                        a=buf_2.get_action(sample_index_2),
                        r=buf_2.get_reward(sample_index_2),
                        other_action=buf_2.get_other_action(sample_index_2),
                        other_action_=np.hstack([a_2_3_,a_2_1_]))
                

            goodag_2.learn(s=buf_3.get_observation(sample_index_3),
                        s_=buf_3.get_next_observation(sample_index_3),
                        a=buf_3.get_action(sample_index_3),
                        r=buf_3.get_reward(sample_index_3),
                        other_action=buf_3.get_other_action(sample_index_3),
                        other_action_=np.hstack([a_3_1_,a_3_2_]))
                


            if noise_rate > NOISE_BOTTOM:
                noise_rate *= 0.9999
            
            if (episode + 1) % 500 == 0:
                saver.save(sess, './simple_adversary_ma_weight/' + str(episode+1) + '.cptk')
                print('episode:{} running_reward:{:.3f} explore:{}'.format(episode+1,running_reward,noise_rate))

            
        observation = observation_
        if (episode + 1) % 200:
            observation = env.reset()