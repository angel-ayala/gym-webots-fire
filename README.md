# Gym Webots Fire Scene
This repository contains the file to use the [OpenAI Gym toolkit](https://github.com/openai/gym) with the [Fire Scene](https://github.com/angel-ayala/webots-fire-scene) simulated in [Webots](https://github.com/cyberbotics/webots).
The environment is intended to train a Reinforcement Learning agent to control a drone for a fire emergency response.
In order to make the drone flight, the algorithm must be capable to work under a continuous domain for both action and state space.
Additionally, the agent's state is represented by an image from which must decide what action to take.
The goal for the agent is to get close of the fire location keeping a safe distance to avoid being burn.

## Action and State space 
The action and state space is represented in a continuous domain where the action is composed with 4 variable: \[phi, theta, psi, thrust\] corresponding to the roll, pitch and yaw angles, and to the altitude which is desired to set the drone.
The angles are in radians and the altitude is in meters.
The 4 values can be set to 0, and the drone will stay in the same position.
If vary the phi or theta (roll, pitch) angle, the drone is capable to move around the environment.
If vary the psi angle (yaw), the drone will be rotated to a desired angle, variated at psi value step, this angle value is \[-pi, pi\].
If vary the thrust value, the drone will be positionated in a desired altitude, variated at thrust value step.

The state space is high-dimensional, represented by the drone's 400x240 pixels BGR channel camera image.
The image is processed to get an image with RGB channels and values in \[0, 1\].

## Reward function
The reward function is the Euclidean distance between the drone's position and the safe zone edge, calculated in the Fire Scene.
The distance can include the altitude difference from the drone or not.
The safe zone edge is defined at the fire location as base, add the radius size and 4 times the fire's height.
This reward function start with a under zero value, and increase while the drone is getting close of the fire location.
If this value is great than zero, the episode's end.

The reward function is included inside the SimController class of the Fire Scene:
```python
class SimController:
    [...]
    def drone_distance(self, fixed=False):    
        # fixed altitude
        if fixed:
            drone_position[1] = fire_position[1] = 0
        else:
            fire_position[1] = 0.5
        
        # Euclidean distance
        delta = drone_position - fire_position
        distance = np.sum(np.square(delta))
        return distance
        
    def compute_reward(self, fixed=True):
        distance = self.drone_distance(fixed)
        risk_zone = self.fire['radius'] + self.fire['height'] *4
        reward = risk_zone - distance
        if reward > 0:
            self._end = True
        
        return reward
    [...]
```
```python    
class ContinuousUavEnv(gym.Env):
    def __init__(self):
        [...]
        self.sim = SimController()
        self.fixed_height = False
        [...]
    [...]  
    def step(self, action):
        self.sim.take_action(action)
        reward = self.sim.compute_reward(self.fixed_height)
        obs = self._get_state()
        self.sim._step()
        return obs, reward, self.sim._end, {}  
    [...]
```

## Considerations of the environment
This environment is an interface for the [Fire Scene](https://github.com/angel-ayala/webots-fire-scene) and take into consideration the [steps to run the Webots scene](https://github.com/angel-ayala/webots-fire-scene#running-the-scene).
The interface use the SimController class to communicate with Webots through its Python API as a [Supervisor Controller](https://www.cyberbotics.com/doc/guide/supervisor-programming).

As mentioned before [here](https://github.com/angel-ayala/webots-fire-scene#running-the-scene), ensure that the WEBOTS_HOME and LD_LIBRARY_PATH OS environment variables are set, and in the PYTHONPATH the Webots lib controller is present.
```
export WEBOTS_HOME=/path/to/webots
export LD_LIBRARY_PATH=$WEBOTS_HOME/lib/controller
export PYTHONPATH=$WEBOTS_HOME/lib/controller/python37
```
After that, ensure of the PYTHONPATH OS environment variables include the folder containing the sim_controller.py file from the Fire Scene.
```
export PYTHONPATH=/path/to/fire/scene/controllers/sim_controller:$PYTHONPATH
```
Notice the 'PYTHONPATH=:$PYTHONPATH' to append the path to a previously defined PYTHONPATH

In order to check:
```
echo $WEBOTS_HOME # must show /path/to/webots
echo $LD_LIBRARY_PATH # must show /path/to/webots/lib/controller
echo $PYTHONPATH # must show  /path/to/fire/scene/controllers/sim_controller:/path/to/webots/lib/controller/python37
```

Finally you can execute your code that implement this environment as usual.