import itertools

import gym
import numpy as np
import tensorflow as tf

from policy_estimator import PolicyEstimator
from value_estimator import ValueEstimator


env = gym.make('CartPole-v1')


discount_factor = 0.99

policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator(discount_factor)


def pick_action_from_probs(probs):
    return np.random.choice(np.arange(len(probs)), p=probs)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i_episode in itertools.count():
        state = env.reset()

        memory = []

        for t in itertools.count():
            env.render()

            action_probs = policy_estimator.predict(sess, state)
            action = pick_action_from_probs(action_probs)

            next_state, reward, done, _ = env.step(action)

            # Update policy estimator
            advantage = value_estimator.td_error(sess, state, reward, next_state)
            # policy_estimator.update(sess, state, advantage, action)

            # Update value estimator
            # value_estimator.update(sess, state, reward, next_state, done)

            memory.append((np.copy(state), action, reward, np.copy(next_state), advantage, done))

            if done:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))

                for transition in reversed(memory):
                    state, action, reward, next_state, advantage, done = transition

                    # Update policy estimator
                    policy_estimator.update(sess, state, advantage, action)

                    # Update value estimator
                    value_estimator.update(sess, state, reward, next_state, done)

                break

            state = next_state
