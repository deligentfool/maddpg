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
BATCH_SIZE = 128
NOISE_BOTTOM = 3

if __name__ == '__main__':
    
    env = make_env.make_env('simple_tag')
    n_ad_feature = 16
    n_ga_feature = 14
    n_action = 2
    n_agent = env.n
    noise_rate = 3
    n_ad_os = 46
    n_ga_os = 48
    sess = tf.Session()

    advag_1 = Maddpg('advag_1_',n_action,n_ad_feature,n_ad_os,n_agent,sess)
    advag_2 = Maddpg('advag_2_',n_action,n_ad_feature,n_ad_os,n_agent,sess)
    advag_3 = Maddpg('advag_3_',n_action,n_ad_feature,n_ad_os,n_agent,sess)
    goodag = Maddpg('goodag_',n_action,n_ga_feature,n_ga_os,n_agent,sess)
    
    saver = tf.train.Saver()
    
    buf_1 = Replay_buffer(n_agent,n_action,n_ad_feature,n_ad_os,CAPACITY,BATCH_SIZE)
    buf_2 = Replay_buffer(n_agent,n_action,n_ad_feature,n_ad_os,CAPACITY,BATCH_SIZE)
    buf_3 = Replay_buffer(n_agent,n_action,n_ad_feature,n_ad_os,CAPACITY,BATCH_SIZE)
    buf_4 = Replay_buffer(n_agent,n_action,n_ga_feature,n_ga_os,CAPACITY,BATCH_SIZE)
    
    sess.run(tf.global_variables_initializer())

    reward_info = np.zeros(1000)
    running_reward = np.sum(reward_info)
    
    observation = env.reset()
    
    for episode in range(MAX_EPISODE):        
        action_1 = advag_1.choose_action(np.array(observation[0])[None,:]) + np.random.randn(2) * noise_rate
        action_2 = advag_2.choose_action(np.array(observation[1])[None,:]) + np.random.randn(2) * noise_rate
        action_3 = advag_3.choose_action(np.array(observation[2])[None,:]) + np.random.randn(2) * noise_rate
        action_4 = goodag.choose_action(np.array(observation[3])[None,:]) + np.random.randn(2) * noise_rate
        
        action = [[0,i[0][0],0,i[0][1],0] for i in [action_1,action_2,action_3,action_4]]
        #action.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
        
        observation_,reward,done,info = env.step(action)
        reward = [i / 5 for i in reward]
        
        reward_info[episode % 1000] = reward[0] + reward[1] + reward[2]
        
        buf_1.add(np.hstack([observation[0],observation[1],observation[2],observation[3]]),np.hstack([action_1,action_2,action_3,action_4]).ravel(),np.hstack([reward[0]]),np.hstack([observation_[0],observation_[1],observation_[2],observation_[3]]))
        buf_2.add(np.hstack([observation[1],observation[2],observation[3],observation[0]]),np.hstack([action_2,action_3,action_4,action_1]).ravel(),np.hstack([reward[1]]),np.hstack([observation_[1],observation_[2],observation_[3],observation_[0]]))
        buf_3.add(np.hstack([observation[2],observation[3],observation[0],observation[1]]),np.hstack([action_3,action_4,action_1,action_2]).ravel(),np.hstack([reward[2]]),np.hstack([observation_[2],observation_[3],observation_[0],observation_[1]]))
        buf_4.add(np.hstack([observation[3],observation[0],observation[1],observation[2]]),np.hstack([action_4,action_1,action_2,action_3]).ravel(),np.hstack([reward[3]]),np.hstack([observation_[3],observation_[0],observation_[1],observation_[2]]))
        
        
        running_reward = 0.9999 * running_reward + 0.0001 * np.sum(reward_info)
        
        if (episode >= START_EPISODE):
            sample_index_1 = buf_1.get_sample_index()
            sample_index_2 = buf_2.get_sample_index()
            sample_index_3 = buf_3.get_sample_index()
            sample_index_4 = buf_4.get_sample_index()
            
            
            a_1_2_ = advag_2.choose_next_action(buf_2.get_self_next_observation(sample_index_1))
            a_1_3_ = advag_3.choose_next_action(buf_3.get_self_next_observation(sample_index_1))
            a_1_4_ = goodag.choose_next_action(buf_4.get_self_next_observation(sample_index_1))
            
            a_2_3_ = advag_3.choose_next_action(buf_3.get_self_next_observation(sample_index_2))
            a_2_4_ = goodag.choose_next_action(buf_4.get_self_next_observation(sample_index_2))
            a_2_1_ = advag_1.choose_next_action(buf_1.get_self_next_observation(sample_index_2))
            
            a_3_4_ = goodag.choose_next_action(buf_4.get_self_next_observation(sample_index_3))
            a_3_1_ = advag_1.choose_next_action(buf_1.get_self_next_observation(sample_index_3))
            a_3_2_ = advag_2.choose_next_action(buf_2.get_self_next_observation(sample_index_3))
            
            a_4_1_ = advag_1.choose_next_action(buf_1.get_self_next_observation(sample_index_4))
            a_4_2_ = advag_2.choose_next_action(buf_2.get_self_next_observation(sample_index_4))
            a_4_3_ = advag_3.choose_next_action(buf_3.get_self_next_observation(sample_index_4))
            
            
            advag_1.learn(s=buf_1.get_self_observation(sample_index_1),
                          s_=buf_1.get_self_next_observation(sample_index_1),
                          a=buf_1.get_self_action(sample_index_1),
                          r=buf_1.get_reward(sample_index_1),
                          other_action=buf_1.get_other_action(sample_index_1),
                          other_action_=np.hstack([a_1_2_,a_1_3_,a_1_4_]),
                          other_S=buf_1.get_other_next_observation(sample_index_1),
                          other_S_=buf_1.get_other_next_observation(sample_index_1))
            
            advag_2.learn(s=buf_2.get_self_observation(sample_index_2),
                          s_=buf_2.get_self_next_observation(sample_index_2),
                          a=buf_2.get_self_action(sample_index_2),
                          r=buf_2.get_reward(sample_index_2),
                          other_action=buf_2.get_other_action(sample_index_2),
                          other_action_=np.hstack([a_2_3_,a_2_4_,a_2_1_]),
                          other_S=buf_2.get_other_next_observation(sample_index_2),
                          other_S_=buf_2.get_other_next_observation(sample_index_2))
            
            advag_3.learn(s=buf_3.get_self_observation(sample_index_3),
                          s_=buf_3.get_self_next_observation(sample_index_3),
                          a=buf_3.get_self_action(sample_index_3),
                          r=buf_3.get_reward(sample_index_3),
                          other_action=buf_3.get_other_action(sample_index_3),
                          other_action_=np.hstack([a_3_1_,a_3_2_,a_3_4_]),
                          other_S=buf_3.get_other_next_observation(sample_index_3),
                          other_S_=buf_3.get_other_next_observation(sample_index_3))
            
            goodag.learn(s=buf_4.get_self_observation(sample_index_4),
                          s_=buf_4.get_self_next_observation(sample_index_4),
                          a=buf_4.get_self_action(sample_index_4),
                          r=buf_4.get_reward(sample_index_4),
                          other_action=buf_4.get_other_action(sample_index_4),
                          other_action_=np.hstack([a_4_1_,a_4_2_,a_4_3_]),
                          other_S=buf_4.get_other_next_observation(sample_index_4),
                          other_S_=buf_4.get_other_next_observation(sample_index_4))
            
            if noise_rate > NOISE_BOTTOM:
                noise_rate *= 0.9999
            
            if (episode + 1) % 500 == 0:
                saver.save(sess, './simple_tag_ma_weight/' + str(episode+1) + '.cptk')
                print('episode:{} running_reward:{:.3f} explore:{:.3f}'.format(episode+1,running_reward,noise_rate))
                
            observation = observation_
        
        if (episode + 1) % 1000:
            observation = env.reset()