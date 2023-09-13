#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:06:18 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

from gym.envs.registration import register

register(
    id='WebotsFireContinuous-v0',
    entry_point='gym_webots_fire.envs:ContinuousUAVEnv',
)

register(
    id='WebotsFireDiscrete-v0',
    entry_point='gym_webots_fire.envs:DiscreteUAVEnv',
)
