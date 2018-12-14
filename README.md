# maddpg

------

This repository is a Multi-Agent Deep Deterministic Policy Gradient implementation with tensorflow. 

The "multiagent-particle-envs-master" is needed.

However, the perform is not very good because the structure of net is not designed closely.

The code structure is as follows.



- simple

A single agent with DDPG for scenario "simple"

- incomplete simple tag

Three adversaries (red) and one good agent (green) for scenario "simple_tag"

MADDPG is applied to the three adversaries and the good agent is a random walker 

* complete simple tag

Three adversaries (red) and one good agent (green) for scenario "simple_tag"

MADDPG is applied to all agents

* simple adversary

one adversary (red), two good agents (green) and 2 landmarks

MADDPG is applied to all agents