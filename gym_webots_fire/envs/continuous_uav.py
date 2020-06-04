#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:11:52 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding

from sim_controller import SimController

class ContinuousUavEnv(gym.Env):
    """Gym enviroment to control the Fire scenario in the Webots simulator."""
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # 'video.frames_per_second' : 30
    }
    
    def __init__(self):
        # Simulation controller
        self.sim = SimController()
        self.speed = self.sim.SIMULATION_MODE_REAL_TIME
        
        # Observation space, the drone's image
        screen_height = self.sim.cam_info[0]
        screen_width = self.sim.cam_info[1]
        self.observation_space = spaces.Box(low=0, high=255, 
                    shape=(screen_height, screen_width, 3), dtype=np.uint8)
        
        # Action space, the angles and altitud
        self.action_space = spaces.Box(self.sim.limits[0], self.sim.limits[1])
        
        self.fixed_height = False
        self.started = False
        self.viewer = None
        self.seed()
     
    def seed(self, seed=None):
        """Set seed for the environment random generations."""
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31. Takem from atari_env.py
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        self.sim.seed(seed2)
        return [seed1, seed2]
    
    def _get_state(self):
        """Process the image to get a nomalized [0, 1] RGB image."""
        img = np.array(self.sim.get_state())
        img = img[:, :, [2, 1, 0]] #BGR2RGB
        img /= 255.
        return img
    
    def step(self, action):
        """Do an action step inside the Webots simulator."""
        self.sim.take_action(action)
        
        reward = self.sim.compute_reward(self.fixed_height)
        obs = self._get_state()
        self.sim._step()
        return obs, reward, self.sim._end, {}
    
    def _reset(self):
        """Reset and stop the Webots simulation."""
        if self.sim.is_running: # episode's end
            self.started = False
            self.sim.restart_environment()
            self.sim.stop_simulation()
        
    def reset(self):
        """Restart the Webots simulation."""
        # stop the simulation         
        self._reset()
        # start the simulation
        self.sim.randomize_fire_position() # Random FireSmoke position
        self.sim.run_simulation(self.speed)
        self.sim._step()
        if not self.started:
            print("Waiting for drone to be ready...")
            while self.sim.getTime() <= 2.5: # 2.5s dead time
                self.sim._step()
            self.started = True
            print("OK")
            
        return self.sim.get_state()
        
    def render(self, mode='human'):
        """Render the environment from Webots simulation."""
        img = self._get_state()
        if mode == 'rgb_array':
            return img        
        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)            
            return self.viewer.isopen
        
    def close(self):
        """Close the environment and stop the simulation."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
        self._reset()
            
    
    
    
        