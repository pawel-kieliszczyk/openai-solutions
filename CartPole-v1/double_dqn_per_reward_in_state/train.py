import collections
import numpy as np
import tensorflow as tf
import gym

from estimator import Estimator
from prioritized_replay_memory import PrioritizedReplayMemory


env = gym.make('CartPole-v1')


num_episodes = 3000

random_action_probability_start = 0.99 # starting random action probability
random_action_probability_end = 0.1 # ending random action probability
random_action_probability_decay = 0.99

discount_factor = 0.99

replay_memory_size = 10000
replay_memory_initial_size = 1000
batch_size = 16

replay_memory = PrioritizedReplayMemory(replay_memory_size)

estimator_1 = Estimator(discount_factor, "estimator_1")
estimator_2 = Estimator(discount_factor, "estimator_2")

recent_timesteps = collections.deque(maxlen=100)

global_step = 0

with tf.Session() as sess:
    random_action_probability = random_action_probability_start
    sess.run(tf.global_variables_initializer())

    # initialize replay memory
    done = True
    current_score = 0.0
    for _ in range(replay_memory_initial_size):
        if done:
            current_score = 0.0
            state = env.reset()
            state = np.append(state, [current_score])
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        current_score += reward
        next_state = np.append(next_state, [current_score])

        replay_memory.add(reward, (state, action, reward, next_state, done))
        state = next_state


    for i_episode in range(num_episodes):
        state = env.reset()
        current_score = 0.0
        state = np.append(state, [current_score])

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

            current_score += reward
            next_state = np.append(next_state, [current_score])

            if global_step % 2 == 0:
                error = estimator_1.td_errors(sess, estimator_2, [state], [action], [reward], [next_state])[0]
                replay_memory.add(error, (state, action, reward, next_state, done))
            else:
                error = estimator_2.td_errors(sess, estimator_1, [state], [action], [reward], [next_state])[0]
                replay_memory.add(error, (state, action, reward, next_state, done))

            samples = replay_memory.sample(batch_size)
            indices_batch, samples_batch = map(np.array, zip(*samples))
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples_batch))

            if global_step % 2 == 0:
                estimator_1.update(sess, estimator_2, states_batch, action_batch, reward_batch, next_states_batch,
                                   done_batch)
                errors = estimator_1.td_errors(sess, estimator_2, states_batch, action_batch, reward_batch,
                                               next_states_batch)
                for i in range(len(indices_batch)):
                    replay_memory.update(indices_batch[i], errors[i])
            else:
                estimator_2.update(sess, estimator_1, states_batch, action_batch, reward_batch, next_states_batch,
                                   done_batch)
                errors = estimator_2.td_errors(sess, estimator_1, states_batch, action_batch, reward_batch,
                                               next_states_batch)
                for i in range(len(indices_batch)):
                    replay_memory.update(indices_batch[i], errors[i])

            global_step += 1

            if done:
                recent_timesteps.append(t+1)
                print("Episode {} finished after {} timesteps (average {})".format(i_episode, t+1 , np.mean(recent_timesteps)))
                break

            state = next_state
