import tensorflow as tf
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

from agent_pgn import Agent

def main():

	continue_training=True # If we want to load checkpoint
	display_environment=True

	env = gym.make('Pong-v0') # Initially make environment
	input_shape = env.observation_space.shape
	pgn = Agent(hidden_layer=200, learning_rate=0.0005, input_shape=input_shape) # Agent Neural Network

	if continue_training:
		print("Continuing Training...")
		pgn.load()

	episode_number = 1 
	running_reward = None 
	gamma = 0.99 # Discount factor
	save_batch_number = 20 # Save after every 20 episodes
	batch_size = 1 # Train after each episode
	batch_memory = []

	while True:
		# Begin new episode/game
		episode_reward = 0
		steps = 0

		last_observation = env.reset()
		last_observation = prepro(last_observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		observation = prepro(observation)

		while not done:

			if display_environment:
				env.render()
			# Take difference between images and preprocess. 
			observation_diff = observation - last_observation
			last_observation = observation
			up_prob = pgn.predict(observation_diff)[0] # Agent prediction
			action = pgn.choose_action(up_prob) # Choose action (up/down)
			observation, reward, done, info = env.step(action) # Implement action and receive new state and reward
			observation = prepro(observation)
			episode_reward += reward
			steps += 1

			# Store state, action and reward for training
			if action == 2:
				batch_memory.append((observation_diff, 1, reward))
			else:
				batch_memory.append((observation_diff, 0, reward))

		if running_reward is None:
			running_reward = episode_reward
		else:
			running_reward = running_reward*gamma + episode_reward*(1-gamma)

		print("Episode {0} Episode reward {1} Running reward {2}".format(episode_number, episode_reward, running_reward))

		# Now to train after each episode
		if episode_number % batch_size == 0:
			observations, actions, rewards = zip(*batch_memory)
			rewards = discount_rewards(rewards, gamma)

			# Standardize rewards
			rewards -= np.mean(rewards)
			rewards /= np.std(rewards)

			batch_memory = list(zip(observations, actions, rewards))
			loss = pgn.train(batch_memory)
			batch_memory = []


		if episode_number % save_batch_number == 0:
			pgn.save()
			print("Saving... Episode {0} Episode reward {1} Running reward {2} Loss {3}".format(episode_number, episode_reward, running_reward, loss))

		episode_number += 1

def prepro(I):
	I = I[35:195]  # we crop the top portion
	I = I[::2, ::2, 0]  # downsample by factor of 2
	I[I == 144] = 0  # erase background (background type 1)
	I[I == 109] = 0  # erase background (background type 2)
	I[I != 0] = 1  # everything else (paddles, ball) just set to 1
	return I.astype(np.float).ravel()
				
def discount_rewards(rewards, discount_factor):
	discounted_rewards = np.zeros_like(rewards)
	for t in range(len(rewards)):
		discounted_reward_sum = 0
		discount = 1
		for k in range(t, len(rewards)):
			# Discount based on how far back the action took place. Action closer to the reward are worth more.
			# Earlier observations/actions weighted less e.g. [0.13263988, 0.13397967, 0.135333, ... 0.9801, 0.99, 1.]
			discounted_reward_sum += rewards[k] * discount
			discount *= discount_factor # Keep discounting the discount factor until we meet a + or - 1.
			if rewards[k] != 0:
				break

		discounted_rewards[t] = discounted_reward_sum
	return discounted_rewards

def play_game(episodes=10):

	env = gym.make('Pong-v4') # Initially make environment
	input_shape = env.observation_space.shape
	pgn = Agent(hidden_layer=200, learning_rate=0.0005, input_shape=input_shape) # Agent Neural Network
	pgn.load()

	for episode in range(episodes):
		score = 0
		last_observation = env.reset()
		last_observation = prepro(last_observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		observation = prepro(observation)

		while not done:
			env.render()
			observation_diff = observation - last_observation
			last_observation = observation
			up_prob = pgn.predict(observation_diff)[0] # Agent prediction
			action = pgn.choose_action(up_prob) # Choose action (up/down)
			observation, reward, done, info = env.step(action) # Implement action and receive new state and reward
			observation = prepro(observation)
			score += reward

		if reward == 1:
			print("Game {0} Won by {1}".format(episode+1, score))
		else:
			print("Game {0} Lost by {1}".format(episode+1, score))


if __name__ == '__main__':
	# main()
	play_game()