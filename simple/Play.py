#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 09:16:02 2018

@author: xuzhiwei
"""

import make_env
from Ddpg import Ddpg
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    
    env = make_env.make_env('simple')
    n_feature = 4
    n_action = 2
    n_agent = 1

    agent = Ddpg(n_action,n_feature)
    
    saver = tf.train.Saver()
    
    agent.sess.run(tf.global_variables_initializer())

    saver.restore(agent.sess,'./weight/208000.cptk')

    observation = env.reset()
    step_count = 0
    for episode in range(100000):        
        while True:
            env.render()
            action_true = agent.choose_action(np.array(observation[0])[None,:])
            
            action = [[0,i[0][0],0,i[0][1],0] for i in [action_true]]
            observation_,reward,done,info = env.step(action)
            
            step_count += 1
            observation = observation_
            
            if (step_count + 1) % 1000 == 0:
                observation = env.reset()
                break