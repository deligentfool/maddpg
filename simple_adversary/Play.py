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
    
    sess.run(tf.global_variables_initializer())

    saver.restore(sess,'./weight/53000.cptk')

    observation = env.reset()
    step_count = 0
    for episode in range(100000):        
        while True:
            env.render()
            action_1 = advag.choose_action(np.array(observation[0])[None,:]) 
            action_2 = goodag_1.choose_action(np.array(observation[1])[None,:])
            action_3 = goodag_2.choose_action(np.array(observation[2])[None,:])
            
            action = [[0,i[0][0],0,i[0][1],0] for i in [action_1,action_2,action_3]]
            observation_,reward,done,info = env.step(action)
            
            step_count += 1
            observation = observation_
            
            if (step_count + 1) % 200 == 0:
                observation = env.reset()
                break