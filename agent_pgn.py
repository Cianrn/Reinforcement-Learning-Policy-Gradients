import tensorflow as tf
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

class Agent:

	def __init__(self, hidden_layer, learning_rate, input_shape):

		self.save_file = './PGN/model.ckpt'
		self.learning_rate = learning_rate
		self.sess = tf.InteractiveSession()
		self.input_size = input_shape[0]*input_shape[1]*input_shape[2]

		self.x = tf.placeholder(tf.float32, [None, 6400]) # Observations
		self.y = tf.placeholder(tf.float32, [None, 1]) # actions + or - 1 for up or down
		self.reward = tf.placeholder(tf.float32, [None, 1]) # reward received 

		layer1 = tf.layers.dense(self.x, units=hidden_layer, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
		self.out = tf.layers.dense(layer1, units=1, activation=tf.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer())

		# Define loss and training operation
		# The aim is to decrease the probability of actions taken in losing rounds and
		# vice versa. This is done through the reward weights. A negative reward will
		# discourage actions taken in this losing round.
		self.loss = tf.losses.log_loss(labels = self.y,
										predictions = self.out,
										weights = self.reward)

		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = optimizer.minimize(self.loss)

		self.sess.run(tf.global_variables_initializer())


	def predict(self, observation):
		observation = observation.reshape([1, -1])
		up_prob = self.sess.run(self.out, feed_dict={self.x: observation})

		return up_prob

	def train(self, batch):

		observations, actions, rewards = zip(*batch) # * basically unpacks positional arguements
		observations = np.vstack(observations)
		actions = np.vstack(actions)
		rewards = np.vstack(rewards)

		_, loss = self.sess.run([self.train_op, self.loss], feed_dict = {self.x: observations,
																		self.y: actions,
																		self.reward: rewards})

		return loss

	def choose_action(self, up_prob):
		if np.random.uniform() < up_prob:
			return 2
		else:
			return 3

	def save(self):
		print("Saving model...")
		saver = tf.train.Saver()
		saver.save(self.sess, self.save_file)

	def load(self):
		saver = tf.train.Saver()
		saver.restore(self.sess, tf.train.latest_checkpoint('./PGN/'))
