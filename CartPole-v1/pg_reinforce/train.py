import itertools

import gym
import numpy as np
import tensorflow as tf

from policy_estimator import PolicyEstimator


env = gym.make('CartPole-v1')


num_episodes = 10000
discount_factor = 0.99

policy_estimator = PolicyEstimator()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i_episode in range(num_episodes):
        state = env.reset()
        episode = []

        for t in itertools.count():
            env.render()

            action_probs = policy_estimator.predict(sess, state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, reward))

            if done:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                break

            state = next_state

        for t, transition in enumerate(episode):
            total_return = sum(discount_factor ** i * t[2] for i, t in enumerate(episode[t:]))
            policy_estimator.update(sess, transition[0], total_return, transition[1])
