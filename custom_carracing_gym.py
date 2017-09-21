import gym
# from gym import spaces
import numpy as np
from scipy.misc import imresize



class RacingGym:
	def __init__(self, env_name='CarRacing-v0', skip_actions=6, num_frames=4, w=42, h=42):
		# env_name: the name of the Open AI Gym environment. By default, CarRacing-v0
		# skip_actions: the number of frames to repeat an action for
		# num_frames: the number of frames to stack in one state
		# w: width of the state input
		# h: height of the state input.
		self.env = gym.make(env_name)
		self.num_frames = num_frames
		self.skip_actions = skip_actions
		self.w = w
		self.h = h

		# Limit the brake, as per explained in
		# https://github.com/oguzelibol/CarRacingA3C, but we also add 'no action'

		if env_name == 'CarRacing-v0':
		# No action, turn right, turn left, accelerate, brake, respectively.
			self.action_space = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 0.8]]
		# In the future, try all the possible (discrete)  actions with MultiDiscrete action space:
		# self.action_space = gym.spaces.MultiDiscrete([[-1, 1], [0,1], [0,1]])

		else:
			self.action_space = range(self.env.action_space.n)

		# For the MultiDiscrete case, we can't use any in-built functions, so:

		# self.action_size = 12

		self.action_size = len(self.action_space)

		self.state = None
		self.env_name = env_name

	# Preprocess the input and stack the frames.
	def preprocess(self, obs, is_start=False):
		# First convert to grayscale
		grayscale = obs.astype('float32').mean(2)
		# Resize the image to w x h and scale to between 0 and 1
		s = imresize(grayscale, (self.w, self.h)).astype('float32')*(1.0/255.0)
		# Next reshape the image to a 4D array with 1st and 4th dimensions of
		# size 1
		s = s.reshape(1, s.shape[0], s.shape[1], 1)
		# Now stack the frames. If this is the first frame, then repeat it
		# num_frames times.
		if is_start or self.state is None:
			self.state = np.repeat(s, self.num_frames, axis = 3)
		else:
			self.state = np.append(s, self.state[:,:,:,:self.num_frames-1], axis = 3)
		return self.state

	# Render the current frame
	def render(self):
		self.env.render()

	# Reset the environment and return the state.
	def reset(self):
		return self.preprocess(self.env.reset(), is_start = True)

	# Step the environment with the given action
	def step(self, action_idx):
		action = self.action_space[action_idx]
		accum_reward = 0
		prev_s = None
		for _ in range( self.skip_actions):
			s, r, term, info = self.env.step(action)
			accum_reward += r
			if term:
				break
			prev_s = s

		if prev_s is not None:
			s = np.maximum.reduce([s, prev_s])
		return self.preprocess(s), accum_reward, term, info

