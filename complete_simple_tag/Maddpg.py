# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:35:04 2018

@author: Xzw

E-mail: diligencexu@gmail.com
"""

import tensorflow as tf
import numpy as np

class Maddpg(object):
    def __init__(self, name, a_dim, s_self_dim, s_other_dim, agent, sess, hidden_unit=128, tau=0.01, gamma=0.9, a_learning_rate=1e-3, c_learning_rate=2e-3):
        self.sess = sess
        self.tau = tau
        self.gamma = gamma
        self.agent = agent
        self.name = name
        self.a_learning_rate = a_learning_rate
        self.c_learning_rate = c_learning_rate
        self.a_dim = a_dim
        self.s_self_dim = s_self_dim
        self.s_other_dim = s_other_dim
        self.hidden_unit = hidden_unit
        
        self.S = tf.placeholder(tf.float32, [None, s_self_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_self_dim], 's_')
        self.other_S = tf.placeholder(tf.float32, [None, s_other_dim], 'other_s')
        self.other_S_ = tf.placeholder(tf.float32, [None, s_other_dim], 'other_s_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.other_action = tf.placeholder(tf.float32,[None,a_dim * (self.agent - 1)])
        self.other_action_ = tf.placeholder(tf.float32,[None,a_dim * (self.agent - 1)])
        
        
        with tf.variable_scope(self.name + 'Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            self.a_ = self._build_a(self.S_, scope='target', trainable=False)
        
        with tf.variable_scope(self.name + 'Critic'):
            q = self._build_c(tf.concat([self.S,self.other_S],-1), 
                              tf.concat([self.a,self.other_action],-1), 
                              scope='eval', 
                              trainable=True)
            q_ = self._build_c(tf.concat([self.S_,self.other_S_],-1), 
                               tf.concat([self.a_,self.other_action_],-1), 
                               scope='target', 
                               trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                           scope=self.name + 'Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                           scope=self.name + 'Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope=self.name + 'Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                           scope=self.name + 'Critic/target')

        self.soft_replace = [tf.assign(t, (1 - self.tau) * t + self.tau * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + self.gamma * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.c_learning_rate).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  
        self.atrain = tf.train.AdamOptimizer(self.a_learning_rate).minimize(a_loss, var_list=self.ae_params)


    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s})

    def learn(self,s,s_,a,r,other_action,other_action_,other_S,other_S_):
        self.sess.run(self.soft_replace)
        self.sess.run(self.atrain, {self.S: s, self.other_action: other_action, self.other_S: other_S})
        self.sess.run(self.ctrain, {self.S: s, self.a: a ,self.other_action: other_action,
                                    self.other_action_: other_action_, self.R: np.array([r]).T,
                                    self.S_: s_, self.other_S: other_S, self.other_S_: other_S_})


    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, self.hidden_unit, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = self.hidden_unit
            w1_s = tf.get_variable('w1_s', [self.s_self_dim + self.s_other_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim * self.agent, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  
        
    def choose_next_action(self,s):
        return self.sess.run(self.a_, {self.S_: s})