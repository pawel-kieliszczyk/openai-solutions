import collections
import numpy as np
import tensorflow as tf
import gym

from estimator import Estimator
from replay_memory import ReplayMemory


env = gym.make('CartPole-v1')


num_episodes = 1000

random_action_probability_start = 0.99 # starting random action probability
random_action_probability_end = 0.01 # ending random action probability
random_action_probability_decay = 0.99

discount_factor = 0.99

replay_memory_size = 10000
batch_size = 16

replay_memory = ReplayMemory(replay_memory_size)

estimator_1 = Estimator(discount_factor, "estimator_1")
estimator_2 = Estimator(discount_factor, "estimator_2")

recent_timesteps = collections.deque(maxlen=100)


global_step = 0

with tf.Session() as sess:
    random_action_probability = random_action_probability_start
    sess.run(tf.global_variables_initializer())

    for i_episode in range(num_episodes):
        state = env.reset()

        for t in range(500):
            env.render()

            action = None
            if np.random.rand(1) < random_action_probability:
                action = env.action_space.sample()
            else:
                if global_step % 2 == 0:
                    action = estimator_1.predict(sess, [state])[0]
                else:
                    action = estimator_2.predict(sess, [state])[0]

            if random_action_probability > random_action_probability_end:
                random_action_probability *= random_action_probability_decay

            next_state, reward, done, _ = env.step(action)

            replay_memory.add(state, action, reward, next_state, done)

            batch_s, batch_a, batch_r, batch_s1, batch_d = replay_memory.get_samples(batch_size)
            if batch_s.shape[0] == batch_size:
                if global_step % 2 == 0:
                    estimator_1.update(sess, estimator_2, batch_s, batch_a, batch_r, batch_s1, batch_d)
                else:
                    estimator_2.update(sess, estimator_1, batch_s, batch_a, batch_r, batch_s1, batch_d)

            global_step += 1

            if done:
                recent_timesteps.append(t+1)
                print("Episode {} finished after {} timesteps (average {})".format(i_episode, t+1 , np.mean(recent_timesteps)))
                break

            state = next_state
