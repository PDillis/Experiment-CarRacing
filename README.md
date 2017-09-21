# Experiment-CarRacing


## Dependencies

We will use the [A3C algorithm](https://arxiv.org/abs/1602.01783) to solve the `CarRacing-v0` environment. The code presented here is based on both [Chris Nicholls](https://github.com/cgnicholls/reinforcement-learning/tree/master/a3c) and [Elibol and Khan's](https://github.com/oguzelibol/CarRacingA3C) implementations, and the architecture of our neural network will be based on [Gym's universe-starter-agent](https://github.com/openai/universe-starter-agent). As such, the dependencies will be the same as the latter, except for the fact that our code will be written specifically for Python 3.5.


## Action and State spaces
The continuity of the action and observation spaces are key characteristics of this Gym environment, which are perhaps what lead to the obscurity of the `Box2D` spaces as they are not as famous as the Atari 2600 games. Indeed, running:

```python
import gym
env = gym.make('CarRacing-v0')
env.action_space
env.action_space
```

we obtain `Box(3,)` and `Box(96,96,3)`, respectively, so the `env.action_space.n` method in the universe-starter-agent does not work for us here. Digging further, the former implies that our actions are of the form `[steer, gas, brake]`, with `steer`, `gas` and `brake` being real numbers. However, not any number may be entered. Indeed, running:

```python
env.action_space.low
env.action_space.high
```

will yield, respectively, `array([-1, 0, 0])` and `array([1, 1, 1])`, so `-1<=steer<=1`, `0<=gas<=1` and `0<=brake<=1`. 

### Discretizing the action space

Due to both computing limitations and some combination of actions making no sense or being 'dangerous' (e.g. gas and braking at the same time, or accelerating and steering as explained in the [documentation](https://github.com/olegklimov/gym/blob/master/gym/envs/box2d/car_racing.py) of the environment), we will discretize the actions available to our agent. Indeed, we can try using the `MultiDiscrete` space like so:

```python
from gym import spaces
self.action_space = spaces.MultiDiscrete([[-1, 1], [0, 1], [0, 1]])
```

However, this would still present us with some 'forbidden' combinations of actions. As such, we will then limit our available actions to 5:

```python
self.action_space = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 0.8]]
```

that is, do nothing, turn right, turn left, accelerate, and brake, respectively, where our brake is limited to `0.8` following the recommendation by [Elibol and Khan's](https://github.com/oguzelibol/CarRacingA3C) implementation. In the future we plan to ignore this limitation and use the `MultiDiscrete` action space, as it is always possible that our agent might find that some combinations of actions which seem nonsensical to us might be of use for specific scenarios.

## Running

To run our agent, we run the following in the command prompt:

```
python a3c_lstm.py -env <env_name> -s <save_path> -t <num_threads> -r <the T in save_path-t>
```

*Note:* this is our first version of the code, and as such it is most likely full of errors, as well as it being a close copy of [Chris Nicholls's](https://github.com/cgnicholls/reinforcement-learning/tree/master/a3c) implementation, as his comments have aided in understanding what goes behind stage.
