import collections
import numpy as np
import tensorflow as tf
import gym

from estimator import Estimator
from replay_memory import ReplayMemory


env = gym.make('CartPole-v0')


num_episodes = 500

random_action_probability_start = 0.99 # starting random action probability
random_action_probability_end = 0.1 # ending random action probability
random_action_probability_decay = 0.99

discount_factor = 0.9

replay_memory_size = 10000
batch_size = 16

update_target_estimator_every_n_episodes = 1

replay_memory = ReplayMemory(replay_memory_size)

q_estimator = Estimator(discount_factor, "q_estimator")
target_estimator = Estimator(discount_factor, "target_estimator")


with tf.Session() as sess:
    random_action_probability = random_action_probability_start
    sess.run(tf.global_variables_initializer())

    for i_episode in range(num_episodes):
        state = env.reset()

        if i_episode % update_target_estimator_every_n_episodes == 0:
            target_estimator.copy_model_from(sess, q_estimator)

        for t in range(500):
            env.render()

            action = None
            if np.random.rand(1) < random_action_probability:
                action = env.action_space.sample()
            else:
                action = q_estimator.predict(sess, [state])[0]

            if random_action_probability > random_action_probability_end:
                random_action_probability *= random_action_probability_decay

            next_state, reward, done, _ = env.step(action)

            replay_memory.add(state, action, reward, next_state, done)

            batch_s, batch_a, batch_r, batch_s1, batch_d = replay_memory.get_samples(batch_size)
            if batch_s.shape[0] == batch_size:
                q_estimator.update(sess, target_estimator, batch_s, batch_a, batch_r, batch_s1, batch_d)

            if done:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                break

            state = next_state
