import itertools

import gym
import numpy as np
import tensorflow as tf

from policy_estimator import PolicyEstimator
from q_estimator import QEstimator


env = gym.make('CartPole-v1')


discount_factor = 0.99

policy_estimator = PolicyEstimator()
q_estimator = QEstimator(discount_factor)


def pick_action_from_probs(probs):
    return np.random.choice(np.arange(len(probs)), p=probs)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i_episode in itertools.count():
        state = env.reset()

        action_probs = policy_estimator.predict(sess, state)
        action = pick_action_from_probs(action_probs)

        for t in itertools.count():
            env.render()

            next_state, reward, done, _ = env.step(action)

            if not done:
                next_action_probs = policy_estimator.predict(sess, next_state)
                next_action = pick_action_from_probs(next_action_probs)

                # Update policy estimator
                advantage = q_estimator.predict(sess, state)[action]
                policy_estimator.update(sess, state, advantage, action)

                # Update value estimator
                q_estimator.update(sess, state, action, reward, next_state, next_action)

                state = next_state
                action = next_action
            else:
                # Update policy estimator
                advantage = q_estimator.predict(sess, state)[action]
                policy_estimator.update(sess, state, advantage, action)

                # Update value estimator
                q_estimator.update_done(sess, state, action, reward)

                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                break
