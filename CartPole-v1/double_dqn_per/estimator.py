import numpy as np
import tensorflow as tf


class Estimator(object):
    def __init__(self, update_discount_factor, name):
        self.discount_factor = update_discount_factor
        self._build_model(name)

    def _build_model(self, name):
        with tf.name_scope(name):
            self.input = tf.placeholder(tf.float32, [None, 4], name="input")
            self.target_output = tf.placeholder(tf.float32, [None, 2], name="target_output")

            with tf.name_scope("hidden1"):
                self.W1 = tf.Variable(tf.truncated_normal([4, 128], stddev=0.01), name="W")
                self.b1 = tf.Variable(tf.zeros([128]), name="b")
                self.hidden1 = tf.nn.tanh(tf.matmul(self.input, self.W1) + self.b1)

            with tf.name_scope("hidden2"):
                self.W2 = tf.Variable(tf.truncated_normal([128, 128], stddev=0.01), name="W")
                self.b2 = tf.Variable(tf.zeros([128]), name="b")
                self.hidden2 = tf.nn.tanh(tf.matmul(self.hidden1, self.W2) + self.b2)

            with tf.name_scope("output"):
                self.W3 = tf.Variable(tf.truncated_normal([128, 2], stddev=0.01), name="W")
                self.b3 = tf.Variable(tf.zeros([2]), name="b")
                self.output = tf.matmul(self.hidden2, self.W3) + self.b3

            with tf.name_scope("predictions"):
                self.predictions = tf.argmax(self.output, 1, name="predictions")

            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.squared_difference(self.output, self.target_output))

                # regularization
                for w in [self.W1, self.W2, self.W3]:
                    self.loss += 0.001 * tf.reduce_sum(tf.square(w))

            with tf.name_scope("train"):
                self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def predict(self, session, states_batch):
        return session.run(self.predictions, feed_dict={self.input: states_batch})

    def td_errors(self, session, second_estimator, states_batch, actions_batch, rewards_batch, next_states_batch):
        batch_size = len(states_batch)
        next_best_actions = session.run(self.predictions, feed_dict={self.input: next_states_batch})

        next_q_values = session.run(second_estimator.output, feed_dict={second_estimator.input: next_states_batch})

        target_q_values = session.run(self.output, feed_dict={self.input: states_batch})

        errors = np.zeros(batch_size)
        for i in range(batch_size):
            errors[i] = abs(target_q_values[i, actions_batch[i]] - rewards_batch[i] - self.discount_factor * next_q_values[i, next_best_actions[i]])

        return errors

    def update(self, session, second_estimator, states_batch, actions_batch, rewards_batch, next_states_batch, done_batch):
        batch_size = states_batch.shape[0]
        next_best_actions = session.run(self.predictions, feed_dict={self.input: next_states_batch})

        next_q_values = session.run(second_estimator.output, feed_dict={second_estimator.input: next_states_batch})

        target_q_values = session.run(self.output, feed_dict={self.input: states_batch})

        for i in range(batch_size):
            if done_batch[i]:
                target_q_values[i, actions_batch[i]] = rewards_batch[i]
            else:
                target_q_values[i, actions_batch[i]] = rewards_batch[i] + self.discount_factor * next_q_values[i, next_best_actions[i]]

        session.run(self.train_step, feed_dict={self.input: states_batch, self.target_output: target_q_values})
