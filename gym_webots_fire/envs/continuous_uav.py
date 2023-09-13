#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:11:52 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""
import gym
import cv2
import numpy as np

from gym import spaces, logger
from gym.utils import seeding

from sim_controller import SimController
from sim_controller import min_max_norm, compute_distance


class ContinuousUAVEnv(gym.Env):
    """Gym enviroment to control the Fire scenario in the Webots simulator."""

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        # 'video.frames_per_second' : 30
    }

    def __init__(self, time_limit=7500,  # 1 min
                 max_no_action_steps=625,  # 5 sec
                 frame_skip=125,  # 1 sec
                 goal_threshold=5., init_altitude=25.,
                 altitude_limits=[11, 75],
                 fire_pos=[-40, 40],
                 fire_dim=[7, 5]):
        # Simulation controller
        logger.info('Checking Webots connection...')
        self.sim = SimController(init_altitude=init_altitude)
        # self.sim.start_simulation(self.sim.SIMULATION_MODE_REAL_TIME)
        # self.sim.stop_simulation()
        self.sim.sync()
        logger.info('Connected to Webots')
        self.speed = self.sim.SIMULATION_MODE_REAL_TIME

        # Action space, the angles and altitud
        self.action_space = spaces.Box(low=self.sim.limits[0],
                                       high=self.sim.limits[1],
                                       shape=(self.sim.limits.shape[-1], ),
                                       dtype=np.float32)
        # Observation space
        width = self.sim.state_shape[1]
        half_height = self.sim.state_shape[0] // 2
        hcenter = width // 2
        self.center_idx = (hcenter - half_height, hcenter + half_height)
        self.resized_shape = (self.sim.state_shape[-1], 84, 84)

        # Observation space, the drone's camera image
        self.obs_type = np.uint8
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=self.resized_shape,
                                            dtype=self.obs_type)

        # time limit
        self._episode_steps = 0  # time limit control
        self._max_episode_steps = time_limit
        self._no_action_steps = 0  # no action control
        self._max_no_action_steps = round(max_no_action_steps / frame_skip)
        self._end = False  # episode end flag
        # runtime vars
        self._frame_skip = frame_skip
        self._reward_lim = [-200 - 50 * (frame_skip - 1), 100 * frame_skip]
        self._goal_distance = 0  # reward helper
        self._fire_pos = fire_pos
        self._fire_dim = fire_dim
        self.goal_threshold = goal_threshold  # **2
        self.altitude_limits = altitude_limits
        self.viewer = None
        self.last_image = None
        self.last_state = None
        self.seed()

    def reset(self):
        """Reset episode in the Webots simulation."""
        # restart simulation
        self.sim.reset_simulation()
        self.sim.sync(True)
        if self._fire_pos:
            self.sim.set_fire_position(self._fire_pos)
            self.sim.set_fire_dim(*self._fire_dim)
        else:
            # randomize fire position
            self.sim.randomize_fire_position()
        self.sim.play_faster()
        self._end = False
        self._episode_steps = 0
        self._goal_distance = 0
        self._no_action_steps = 0

        # wait to lift the drone
        logger.info("Lifting the drone...")
        while not self.sim.drone_lifted:  # dead time
            self._get_state()  # clear state buffer
            self.sim._step()

        logger.info("Drone lifted")
        self.last_state = self._get_state()
        self._last_drone_pos = self.sim.get_drone_pos()

        return self.last_state[0]

    def seed(self, seed=None):
        """Set seed for the environment random generations."""
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31. Takem from atari_env.py
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        self.sim.seed(seed2)
        return [seed1, seed2]

    def preprocess_image(self, img):
        """Resize and normalize the state."""
        # make it square from center
        result = np.asarray(img[:, self.center_idx[0]:self.center_idx[1], :],
                            dtype=self.obs_type)
        # resize
        shape = (self.resized_shape[1], self.resized_shape[2])
        result = cv2.resize(result, shape, interpolation=cv2.INTER_AREA)
        # channel first
        result = np.swapaxes(result, 2, 0)
        # normalize
        # result /= 255.
        return result

    def _get_state(self):
        """Process the image to get a RGB image."""
        img, sensors, angles, north_deg = self.sim.get_state()
        img = img[:, :, [2, 1, 0]]  # BGR2RGB
        self.last_image = img.copy()
        img = self.preprocess_image(img)

        return img, sensors, angles, north_deg

    def constrained_action(self, action, obs):
        """Check drone position and orientation to keep inside FlightArea."""
        # clip action values
        action = np.clip(action, self.sim.limits[0], self.sim.limits[1])

        roll_angle, pitch_angle, yaw_angle, altitude = action

        # check world contraints
        out_alt = self.sim.check_altitude(self.altitude_limits)
        out_area = self.sim.check_flight_area()
        north_deg = obs[3]

        orientation = [north_deg >= 90 and north_deg < 270,   # north
                       north_deg < 90 or north_deg >= 270,    # south
                       north_deg < 180 and north_deg >= 0,    # east
                       north_deg >= 180 and north_deg < 360]  # west
        movement = [pitch_angle > 0,  # north
                    pitch_angle < 0,  # south
                    roll_angle > 0,   # east
                    roll_angle < 0]   # west

        if out_area[0]:
            if ((orientation[0] and movement[0])
                    or (orientation[1] and movement[1])):  # N,S
                pitch_angle = 0.

            if ((orientation[2] and movement[3])
                    or (orientation[3] and movement[2])):  # E,W
                roll_angle = 0.

        if out_area[1]:
            if ((orientation[0] and movement[1])
                    or (orientation[1] and movement[0])):  # N,S
                pitch_angle = 0.

            if ((orientation[2] and movement[2])
                    or (orientation[3] and movement[3])):  # E,W
                roll_angle = 0.

        if out_area[2]:
            if ((orientation[0] and movement[2])
                    or (orientation[1] and movement[3])):  # N,S
                roll_angle = 0.

            if ((orientation[2] and movement[0])
                    or (orientation[3] and movement[1])):  # E,W
                pitch_angle = 0.

        if out_area[3]:
            if ((orientation[0] and movement[3])
                    or (orientation[1] and movement[2])):  # N,S
                roll_angle = 0.

            if ((orientation[2] and movement[1])
                    or (orientation[3] and movement[0])):  # E,W
                pitch_angle = 0.

        if ((out_alt[0] and altitude > 0)  # ascense
                or (out_alt[1] and altitude < 0)):  # descense
            altitude = 0.

        return roll_angle, pitch_angle, yaw_angle, altitude

    def compute_reward(self, obs):
        """Compute the distance-based reward.

        Compute the distance between drone and fire.
        This consider a risk_zone to 4 times the fire height as mentioned in
        Firefighter Safety Zones: A Theoretical Model Based on Radiative
        Heating, Butler, 1998.

        :param float distance_threshold: Indicate the acceptable distance
            margin before the fire's risk zone.
        """
        img, sensors, angles, north_deg = obs
        goal_threshold = self.sim.risk_distance + self.goal_threshold
        goal_distance = self.sim.get_goal_distance()

        reward = -1.0 + self._goal_distance - goal_distance

        # TODO: check drone collision
        # reward = -50

        wrong_altitude = self.sim.check_altitude(self.altitude_limits)
        outside_area = self.sim.check_flight_area()
        is_flipped = self.sim.check_flipped(angles)
        object_near = self.sim.check_near_object(sensors)

        # terminal states
        if is_flipped:
            # drone's propellers side up
            logger.info("Terminal state reached, flipped")
            reward = -100
            self._end = True
        elif goal_distance < self.sim.risk_distance:
            # risk zone achieved
            logger.info("Terminal state reached, risk zone")
            reward = -200
            self._end = True

        # not terminal, must be avoided
        if any(outside_area) or any(wrong_altitude):
            reward = -50
        elif any(object_near):
            reward = -25
        elif goal_distance < goal_threshold:
            # goal achieved
            reward = 100
            # reset no_action_step for infinite loop acception no_action to
            # keep the current position
            # TODO: check if is looking the fire
            # self._no_action_steps = -1

        self._goal_distance = goal_distance
        return reward

    def step(self, action):
        """Perform an action step in the Webots scene."""
        # assert self.action_space.contains(action)
        reward = 0
        # action repeat
        for _ in range(self._frame_skip):
            action = self.constrained_action(action, self.last_state)
            self.sim.take_action(action)  # perform action
            observation = self._get_state()  # read state
            reward += self.compute_reward(observation)  # obtain reward
            if self._end:
                break

        # time limit control
        self._episode_steps += 1
        if self._episode_steps >= self._max_episode_steps:
            logger.info("Terminal state reached, time limit")
            self._end = True

        # no action control
        _current_drone_pos = self.sim.get_drone_pos()
        _diff_pos = compute_distance(_current_drone_pos, self._last_drone_pos)
        self._no_action_steps = self._no_action_steps + 1 \
            if _diff_pos <= 0.01 else 0

        if self._no_action_steps >= self._max_no_action_steps:
            logger.info("Terminal state reached, max no action steps")
            reward = -25
            self._end = True

        self._last_drone_pos = _current_drone_pos
        self.last_state = observation

        # normalize reward
        reward = min_max_norm(reward, a=-1, b=1,
                              minx=self._reward_lim[0],
                              maxx=self._reward_lim[1])
        # return image only
        obs = observation[0]

        # process reward
        # reward = self.sparse_reward(reward)

        return obs, reward, self._end, {}

    def render(self, mode='human'):
        """Render the environment from Webots simulation."""
        if mode == 'rgb_array':
            return self.last_image

        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()

            if self.last_image is not None:
                self.viewer.imshow(self.last_image)

            return self.viewer.isopen

    def close(self):
        """Close the environment and stop the simulation."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
