#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:16:02 2018

@author: xuzhiwei
"""

import make_env
from Maddpg import Maddpg
import tensorflow as tf
import numpy as np

START_EPISODE = 5000
MAX_EPISODE = 10000000
CAPACITY = 1000000
BATCH_SIZE = 128

if __name__ == '__main__':
    
    env = make_env.make_env('simple_tag')
    n_ad_feature = 16
    n_ga_feature = 14
    n_action = 2
    n_agent = env.n
    noise_rate = 1
    n_ad_os = 46
    n_ga_os = 48
    sess = tf.Session()

    advag_1 = Maddpg('advag_1_',n_action,n_ad_feature,n_ad_os,n_agent,sess)
    advag_2 = Maddpg('advag_2_',n_action,n_ad_feature,n_ad_os,n_agent,sess)
    advag_3 = Maddpg('advag_3_',n_action,n_ad_feature,n_ad_os,n_agent,sess)
    goodag = Maddpg('goodag_',n_action,n_ga_feature,n_ga_os,n_agent,sess)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())

    saver.restore(sess,'./weight/174000.cptk')
    observation = env.reset()
    step_count = 0
    for episode in range(100000):    
        while True:
            env.render()
            
            action_1 = advag_1.choose_action(np.array(observation[0])[None,:])
            action_2 = advag_2.choose_action(np.array(observation[1])[None,:])
            action_3 = advag_3.choose_action(np.array(observation[2])[None,:])
            action_4 = goodag.choose_action(np.array(observation[3])[None,:])
            
            action = [[0,i[0][0],0,i[0][1],0] for i in [action_1,action_2,action_3,action_4]]
            
            observation_,reward,done,info = env.step(action)
            step_count += 1
            
            observation = observation_
            if (step_count + 1) % 1000 == 0:
                observation = env.reset()
                break