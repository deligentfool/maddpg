#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:12:26 2018

@author: xuzhiwei
"""

import numpy as np

class Replay_buffer(object):
    def __init__(self,n_agent,n_action,n_feature,capacity,batch_size):
        self.n_agent = n_agent
        self.n_action = n_action
        self.n_feature = n_feature
        self.capacity = capacity
        self.batch_size = batch_size
        self.store_pos = 0
        self.counter = 0

        self.buf = np.zeros([self.capacity,self.n_feature * 2 * n_agent + self.n_action * self.n_agent + 1])
        
    def get_sample_index(self):
        if self.counter < self.capacity:
            sample_index = np.random.choice(np.arange(self.counter),self.batch_size)
        else:
            sample_index = np.random.choice(np.arange(self.capacity),self.batch_size)
        return sample_index
    
    def get_self_action(self,sample_index):
        return self.buf[sample_index,self.n_feature * 2 * self.n_agent:self.n_feature * 2 * self.n_agent + self.n_action]
    
    def get_other_action(self,sample_index):
        return self.buf[sample_index,self.n_feature * 2 * self.n_agent + self.n_action:-1]
    
    def get_all_action(self,sample_index):
        return self.buf[sample_index,self.n_feature * 2 * self.n_agent:-1]
    
    def get_reward(self,sample_index):
        return self.buf[sample_index,-1]
    
    def get_self_observation(self,sample_index):
        return self.buf[sample_index,:self.n_feature]
    
    def get_other_observation(self,sample_index):
        return self.buf[sample_index,self.n_feature:self.n_feature * self.n_agent]
    
    def get_self_next_observation(self,sample_index):
        return self.buf[sample_index,self.n_feature * self.n_agent:self.n_feature * (self.n_agent + 1)]
    
    def get_other_next_observation(self,sample_index):
        return self.buf[sample_index,self.n_feature * (self.n_agent + 1):self.n_feature * 2 * self.n_agent]
    
    def get_all_observation(self,sample_index):
        return self.buf[sample_index,:self.n_feature * self.n_agent]
    
    def get_all_next_observation(self,sample_index):
        return self.buf[sample_index,self.n_feature * 2:self.n_feature * 2 * self.n_agent]
    
    def add(self,s,a,r,s_):
        self.buf[self.store_pos] = np.hstack([s,s_,a,r])
        self.store_pos += 1
        self.store_pos = self.store_pos % self.capacity
        self.counter += 1
    
    def clear(self):
        self.buf = np.zeros([self.capacity,self.n_feature * 2 + self.n_action * self.n_agent + 1])
        self.counter = 0
        self.store_pos = 0
        
    