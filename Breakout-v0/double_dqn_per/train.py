import collections
import numpy as np
import tensorflow as tf
import gym

from estimator import Estimator
from prioritized_replay_memory import PrioritizedReplayMemory
from state_processor import StateProcessor


env = gym.make('Breakout-v0')


num_episodes = 10000

random_action_probability_start = 0.999 # starting random action probability
random_action_probability_end = 0.1 # ending random action probability
random_action_probability_decay = 0.9995

discount_factor = 0.99

replay_memory_size = 500000
replay_memory_initial_size = 50000
batch_size = 32

state_processor = StateProcessor()

replay_memory = PrioritizedReplayMemory(replay_memory_size)

estimator_1 = Estimator(discount_factor, "estimator_1")
estimator_2 = Estimator(discount_factor, "estimator_2")

recent_timesteps = collections.deque(maxlen=100)

global_step = 0

with tf.Session() as sess:
    random_action_probability = random_action_probability_start
    sess.run(tf.global_variables_initializer())

    # initialize replay memory
    print("Initializing replay memory")
    done = True
    state = None
    for _ in range(replay_memory_initial_size):
        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

        replay_memory.add(reward, (state, action, reward, next_state, done))
        state = next_state
    print("Replay memory initialized")


    for i_episode in range(num_episodes):
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)

        if random_action_probability > random_action_probability_end:
            random_action_probability *= random_action_probability_decay

        for t in range(999999):
            env.render()

            action = None
            if np.random.rand(1) < random_action_probability:
                action = env.action_space.sample()
            else:
                if global_step % 2 == 0:
                    action = estimator_1.predict(sess, [state])[0]
                else:
                    action = estimator_2.predict(sess, [state])[0]

            next_state, reward, done, _ = env.step(action)
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

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
