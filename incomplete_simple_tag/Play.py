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
    
    env = make_env.make_env('simple_tag')
    n_feature = 16
    n_action = 2
    n_agent = env.n - 1
    
    sess = tf.Session()

    agent_1 = Maddpg('agent_1_',n_action,n_feature,n_agent,sess)
    agent_2 = Maddpg('agent_2_',n_action,n_feature,n_agent,sess)
    agent_3 = Maddpg('agent_3_',n_action,n_feature,n_agent,sess)
        
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())

    saver.restore(sess,'./weight/180000.cptk')
    observation = env.reset()
    step_count = 0
    for episode in range(100000):    
        while True:
            env.render()
            
            action_1 = agent_1.choose_action(np.hstack(observation[0])[None,:])
            action_2 = agent_2.choose_action(np.hstack(observation[1])[None,:])
            action_3 = agent_3.choose_action(np.hstack(observation[2])[None,:])
            
            action = [[0,i[0][0],0,i[0][1],0] for i in [action_1,action_2,action_3]]
            action.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            
            observation_,reward,done,info = env.step(action)
            step_count += 1
            
            observation = observation_
            if (step_count + 1) % 1000 == 0:
                observation = env.reset()
                break