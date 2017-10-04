import tensorflow as tf
import numpy as np
from agent_lstm import Agent
import matplotlib.pyplot as plt
import gym
from custom_carracing_gym import RacingGym
import random


def test_network():
	# Reset the graph to make sure we get a new session.
	with tf.Session() as sess:
		# Create an agent
		agent = Agent(session = sess,
		              action_size = 3,
		              optimizer = tf.train.AdamOptimizer(1e-4))

		# Initialize all variables and then check the output of the network
		sess.run(tf.global_variables_initializer())

		print(agent.layers)
		trainable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'network')
		print(trainable_variables)

		trainable_variables_values = sess.run(trainable_variables)
		print("Trainable variables initial values: ")
		print([var for var in trainable_variables_values])

		print("Shapes of trainable variables: ")
		print([np.shape(var) for var in trainable_variables_values])

		print("Total number of variables: ")
		print(np.sum([np.prod(np.shape(var)) for var in trainable_variables_values]))


def display_output(sess, agent, state):
	layer_outputs = {}
	for k, v in agent.layers:
		layer_outputs[k] = sess.run(v, feed_dict = {agent.state: state})

	print(layer_outputs)


def test_network_output():
	# Reset the graph to make sure we get a new session.
	tf.reset_default_graph()
	with tf.Session() as sess:
		# Create an agent
		agent = Agent(session = sess, action_size = 5, optimizer = tf.train.AdamOptimizer(1e-4))

		# Initialize al variables and then check the output of the network
		sess.run(tf.global_variables_initializer())

		# Get a random state from
		state = get_random_state('CarRacing-v0')

		# Plot the images indexed by the fourth dimension. The first dimension should be size 1,
		# so the image is really 3D, and we show each image indexed by the last dimension.
		assert np.shape(state)[0] == 1
		plot_3d(state[0, :, :, :], title = 'Initial state')

		# We can also plot the output of the first conv layer
		conv1_output = sess.run(agent.layers['conv1'], feed_dict = {agent.state: state})
		plot_3d(conv1_output[0, :, :, :], title = 'conv1')

		# And plot the output of the second conv layer
		conv2_output = sess.run(agent.layers['conv2'], feed_dict = {agent.state: state})
		plot_3d(conv2_output[0, :, :, :], title = 'conv2')


# Plots each filter in a 3D array by using the third dimension as an index to each 2D grayscale image
def plot_3d(arr, title = None):
	# Show all the filters one by one
	for i in range(np.shape(arr)[-1]):
		if title:
			plt.title(title + ' filter ' + str(i))
		plt.imshow(arr[:, :, i], cmap = 'gray')
		plt.show()


def get_random_state(env_name, random_steps = 20):
	gym_env = gym.make(env_name)
	env = RacingGym(gym_env, 'CarRacing-v0')

	# Reset the environment and then take 20 random actions
	state = env.reset()
	for _ in range(random_steps):
		state, _, _, _ = env.step(random.randrange(env.action_size))

	return state

test_network()
test_network_output()
