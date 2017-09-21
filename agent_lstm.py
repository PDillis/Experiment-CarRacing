import tensorflow as tf
import numpy as np


class Agent:
	def __init__(self, session, action_size, model='mnih-lstm', optimizer = tf.train.AdamOptimizer(1e-4)):
		# session: the tensorflow session
		# action_size: the number of actions
		self.action_size = action_size
		self.optimizer = optimizer
		self.sess = session

		with tf.variable_scope('network'):
			# Placeholders for the action, advantage and target value
			self.action = tf.placeholder('float32',
			                             [None, self.action_size],
			                             name = 'action')
			self.advantages = tf.placeholder('float32',
			                                 [None],
			                                 name ='advantages')
			self.target_value = tf.placeholder('float32',
			                                   [None],
			                                   name = 'target_value')
			# Store the state, policy and value for the network
			if model == 'mnih':
				self.state, self.policy, self.value = self.build_model(84,84,4)
			elif model == 'minh-lstm':
				self.state, self.policy, self.value = self.build_model_lstm(84, 84)
			else:
				# Assume we wanted a feedforward neural network
				self.state, self.policy, self.value = self.build_model_feedforward(4)

			# Get the weights for the network
			self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'network')

		with tf.variable_scope('optimizer'):
			# Compute the one hot vectors for each action given.
			action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)

			# Clip the policy output to avoid zeroes and ones -- these don't play well with the log
			min_policy = 1e-8
			max_policy = 1.0 - 1e-8
			self.log_policy = tf.log(tf.clip_by_value(self.policy, 1e-6, 1-1e-6))

			# For a given state and action, compute the log of the policy at that action for that
			# state. This also works on batches.

			self.log_pi_for_action = tf.reduce_sum(tf.multiply(self.log_policy, action_one_hot), reduction_indices = 1)

			# Takes in R_t - V(S_t) as in the A3C paper. Note that we feed in the advantages so that
			# V(S_t) is treated as a constant for the gradient. This is because V(S_t) is the base-
			# line (called 'b' in the REINFORCE algorithm). As long as the baseline is constant w.r.t.
			# the parameters we are optimizing (in this case those for the policy), then the expected
			# value of grad_theta log pi * b is zero, so the choice of b doesn't affect the expecta-
			# tion, but it does reduce the variance.

			# We want to do gradient ascent on the expected discounted reward. The gradient of the
			# expected discounted reward is the gradient of log pi * (R - estimated V), where R is
			# the sampled reward from the given state following the policy pi. Since we want to
			# maximise this, we define the policy loss as the negative and get tensorflow to do the
			# automatic differentiation for us.

			self.policy_loss = -tf.reduce_mean(self.log_pi_for_action * self.advantages)

			# The value loss is much easier to understand: we want our value function to accurately
			# estimate the sampled discounted rewards, so we just impose a square error loss. Note
			# that the target value should be the discounted reward for the state as just sampled.

			self.value_loss = tf.reduce_mean(tf.square(self.target_value - self.value))

			# Following Mnih's paper, we introduce the entropy as another loss to the policy. The
			# entropy of a probability distribution is just the expected value of - log P(X), denoted
			# E(-log P(X)), which we can compute for our policy at any given state with the following:
			# sum(policy * - log(policy)), as below. This will be a positive number, since self.policy
			# contains numbers between 0 and 1, so the log is negative. Note that entropy is smaller
			# when the probability distribution is more concentrated on one action, so a larger entropy
			# implies more exploration. Thus, we penalise small entropy, or equivalently, add -entropy
			# to our loss.

			self.entropy = tf.reduce_sum(tf.multiply(self.policy, -self.log_policy))

			# Try to minimise the loss. There is some rationale for choosing the weighted linear combina-
			# tion used next, but more research is granted. Note the negative entropy term, which encourages
			# exploration: higher entropy corresponds to less certainty (determinism).

			self.loss = 0.5 * self.value_loss + self.policy_loss - 0.01 * self.entropy

			# Compute the gradient of the loss with respect to all the weights, and create a list of tuples
			# consisting of the gradient to apply to the weight.

			grads = tf.gradients(self.loss, self.weights)
			grads, _ = tf.clip_by_global_norm(grads, 40.0)
			grads_vars = list(zip(grads, self.weights))

			# Create an operator to apply the gradients using the optimizer. Note that apply_gradients is
			# the second part of minimize() for the optimizer, so will minimize the loss.

			self.train_op = optimizer.apply_gradients(grads_vars)

	# We define some helper functions for getting the policy, value and training
	def get_policy(self, state):
		return self.sess.run(self.policy, {self.state: state}).flatten()

	def get_value(self, state):
		return self.sess.run(self.value, {self.state: state}).flatten()

	def get_policy_and_value(self, state):
		policy, value = self.ses.run([self.policy, self.value], {self.state: state})
		return policy.flatten(), value.flatten()

	# Train the network on the given states and rewards.

	def train(self, states, actions, target_values, advantages):
		# Training
		self.sess.run(self.train_op, feed_dict = {self.state: states,
		                                          self.action: actions,
		                                          self. target_value: target_values,
		                                          self.advantages: advantages})

	# Builds the model as in universe-starter-agent, but we get a softmax output for the
	# policy from fc1 and a linear output for the value from fc1.

	def build_model(self, h, w, channels):
		state = tf.placeholder('float32', shape=(None, h, w, channels), name = 'state')

		# We have four convolutional layers as in universe-starter-agent.
		with tf.variable_scope('conv1'):
			conv1 = tf.contrib.layers.convolution2d(inputs = state,
			                                        num_outputs = 32,
			                                        kernel_size = [3, 3],
			                                        stride = [2, 2],
			                                        padding = "SAME",
			                                        activation_fn = tf.nn.elu,
			                                        weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
			                                        biases_initializer = tf.zeros_initializer())

		with tf.variable_scope('conv2'):
			conv2 = tf.contrib.layers.convolution2d(inputs = conv1,
			                                        num_outputs = 32,
			                                        kernel_size = [3, 3],
			                                        stride = [2, 2],
			                                        padding = "SAME",
			                                        activation_fn = tf.nn.elu,
			                                        weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
			                                        biases_initializer = tf.zeros_initializer())

		with tf.variable_scope('conv3'):
			conv3 = tf.contrib.layers.convolution2d(inputs = conv2,
			                                        num_outputs = 32,
			                                        kernel_size = [3, 3],
			                                        stride = [2, 2],
			                                        padding = "SAME",
			                                        activation_fn = tf.nn.elu,
			                                        weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
			                                        biases_initializer = tf.zeros_initializer())

		with tf.variable_scope('conv4'):
			conv4 = tf.contrib.layers.convolution2d(inputs = conv3,
			                                         num_outputs = 32,
			                                         kernel_size = [3, 3],
			                                         stride = [2, 2],
			                                         padding = "SAME",
			                                         activation_fn = tf.nn.elu,
			                                         weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
			                                         biases_initializer = tf.zeros_initializer())

		# Flatten the network
		with tf.variable_scope('flatten'):
			flatten = tf.contrib.layers.flatten(inputs = conv4)
			# Fully connected layer with 256 hidden units
		with tf.variable_scope('fcl1'):
			fcl1 = tf.contrib.layers.fully_connected(inputs = flatten,
			                                         num_outputs = 256,
			                                         activation_fn = tf.nn.elu,
			                                         weights_initializer = tf.contrib.layers.xavier_initializer(),
			                                         biases_initializer = tf.zeros_initializer())
		# FUTURE: Add another FCL
		# The policy output

		with tf.variable_scope('policy'):
			policy = tf.contrib.layers.fully_connected(inputs = fcl1,
			                                           num_outputs = self.action_size,
			                                           activation_fn = tf.nn.softmax,
			                                           weights_initializer = tf.contrib.layers.xavier_initializer(),
			                                           biases_initializer = None)
		# The value output

		with tf.variable_scope('value'):
			value = tf.contrib.layers.fully_connected(inputs = fcl1,
			                                          num_outputs = 1,
			                                          activation_fn = None,
			                                          weights_initializer = tf.contrib.layers.xavier_initializer(),
			                                          biases_initializer = None)

		return state, policy, value


	def build_model_lstm(self, h, w):
		self.layers = {}

		# The state has shape batch size x h x w x 1. We need four dimensions in order to do convolutions

		state = tf.placeholder('float32', shape = (None, h, w, 1), name = 'state')
		self.layers['state'] = state


		# Convolutional layers

		with tf.variable_scope('conv1'):
			conv1 = tf.contrib.layers.convolution2d(inputs = state,
			                                        num_outputs = 32,
			                                        kernel_size = [3, 3],
			                                        stride = [2, 2],
			                                        padding = "VALID",
			                                        activation_fn = tf.nn.elu,
			                                        weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
			                                        biases_initializer = tf.zeros_initializer())
			self.layers['conv1'] = conv1

		with tf.variable_scope('conv2'):
			conv2 = tf.contrib.layers.convolution2d(inputs = conv1,
			                                        num_outputs = 32,
			                                        kernel_size = [3, 3],
			                                        stride = [2, 2],
			                                        padding = "VALID",
			                                        activation_fn = tf.nn.elu,
			                                        weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
			                                        biases_initializer = tf.zeros_initializer())
			self.layers['conv2'] = conv2

		with tf.variable_scope('conv3'):
			conv3 = tf.contrib.layers.convolution2d(inputs = conv2,
			                                        num_outputs = 32,
			                                        kernel_size = [3, 3],
			                                        stride = [2, 2],
			                                        padding = "VALID",
			                                        activation_fn = tf.nn.elu,
			                                        weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
			                                        biases_initializer = tf.zeros_initializer())
			self.layers['conv3'] = conv3

		with tf.variable_scope('conv4'):
			conv4 = tf.contrib.layers.convolution2d(inputs = conv3,
			                                        num_outputs = 32,
			                                        kernel_size = [3, 3],
			                                        stride = [2, 2],
			                                        padding = "VALID",
			                                        activation_fn = tf.nn.elu,
			                                        weights_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
			                                        biases_initializer = tf.zeros_initializer())
			self.layers['conv4'] = conv4

		# Flatten the network

		with tf.variable_scope('flatten'):
			flatten = tf.contrib.layers.flatten(inputs = conv4)
			self.layers['flatten'] = flatten
			# Fully connected layer with 256 hidden units

		with tf.variable_scope('fcl1'):
			fcl1 = tf.contrib.layers.fully_connected(inputs = flatten,
			                                         num_outputs = 256,
			                                         activation_fn = tf.nn.elu,
			                                         weights_initializer = tf.contrib.layers.xavier_initializer(),
			                                         biases_initializer = tf.zeros_initializer())
			self.layers['fcl1'] = fcl1

		# Future: add one more fully connected layer

		with tf.variable_scope('lstm'):
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple = True)
			c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
			h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
			self.rnn_state_init = [c_init, h_init]
			c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c], "c_in")
			h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h], "h_in")
			self.rnn_state_in = (c_in, h_in)
			rnn_in = tf.expand_dims(fcl1, [0])

			# The sequence length is the size of the batch

			sequence_length = tf.shape(state)[:1]
			rnn_state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
			lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell,
			                                             rnn_in,
			                                             initial_state = rnn_state_in,
			                                             sequence_length = sequence_length,
			                                             time_major = False)
			lstm_c, lstm_h = lstm_state
			self.rnn_state_out = (lstm_c[:1,:], lstm_h[:1,:])
			rnn_out = tf.reshape(lstm_outputs, [-1, 256])
			self.layers['rnn_state_init'] = self.rnn_state_init
			self.layers['rnn_state_in'] = self.rnn_state_in
			self.layers['rnn_state_out'] = self.rnn_state_out

		# The policy output

		with tf.variable_scope('policy'):
			policy = tf.contrib.layers.fully_connected(inputs = rnn_out,
			                                           num_outputs = self.action_size,
			                                           activation_fn = tf.nn.softmax,
			                                           weights_initializer = tf.contrib.layers.xavier_initializer(),
			                                           biases_initializer = None)
			self.layers['policy'] = policy

		# The value output

		with tf.variable_scope('value'):
			value = tf.contrib.layers.fully_connected(inputs = rnn_out,
			                                          num_outputs = 1,
			                                          activation_fn = None,
			                                          weights_initializer = tf.contrib.layers.xavier_initializer(),
			                                          biases_initializer = None)
			self.layers['value'] = value

		return state, policy, value

	def build_model_feedforward(self, input_dim, num_hidden = 30):
		self.layers = {}
		state = tf.placeholder('float32', shape = (None, input_dim), name = 'state')

		self.layers['state'] = state


		# Two fully connected layer with num_hidden hidden units
		with tf.variable_scope('fcl1'):
			fcl1 = tf.contrib.layers.fully_connected(inputs = state,
			                                         num_outputs = num_hidden,
			                                         activation_fn = tf.nn.elu,
			                                         weights_initializer = tf.contrib.layers.xavier_initializer(),
			                                         biases_initializer = tf.zeros_initializer())
			self.layers['fcl1'] = fcl1

		with tf.variable_scope('fcl2'):
			fcl2 = tf.contrib.layers.fully_connected(inputs = fcl1,
			                                         num_outputs = num_hidden,
			                                         activation_fn = tf.nn.elu,
			                                         weights_initializer = tf.contrib.layers.xavier_initializer(),
			                                         biases_initializer = tf.zeros_initializer())
			self.layers['fcl2'] = fcl2

		# The policy output to the possible actions
		with tf.variable_scope('policy'):
			policy = tf.contrib.layers.fully_connected(inputs = fcl2,
			                                           num_outputs = self.action_size,
			                                           activation_fn = tf.nn.softmax,
			                                           weights_initializer = tf.contrib.layers.xavier_initializer(),
			                                           biases_initializer = tf.zeros_initializer())
			self.layers['policy'] = policy

		# The value output
		with tf.variable_scope('value'):
			value = tf.contrib.layers.fully_connected(inputs = fcl2,
			                                          num_outputs = 1,
			                                          activation_fn = None,
			                                          weights_initializer = tf.contrib.layers.xavier_initializer(),
			                                          biases_initializer = tf.zeros_initializer())
			self.layers['value'] = value

		return state, policy, value
