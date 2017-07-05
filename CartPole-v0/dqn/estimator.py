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

    def update(self, session, target_estimator, states_batch, actions_batch, rewards_batch, next_states_batch, done_batch):
        batch_size = states_batch.shape[0]

        next_q_values = session.run(target_estimator.output, feed_dict={target_estimator.input: next_states_batch})
        max_target_q_values = next_q_values.max(axis=1)

        target_q_values = session.run(self.output, feed_dict={self.input: states_batch})

        for i in range(batch_size):
            if done_batch[i]:
                target_q_values[i, actions_batch[i]] = rewards_batch[i]
            else:
                target_q_values[i, actions_batch[i]] = rewards_batch[i] + self.discount_factor * max_target_q_values[i]

        session.run(self.train_step, feed_dict={self.input: states_batch, self.target_output: target_q_values})

    def copy_model_from(self, session, other):
        new_W1 = tf.assign(self.W1, other.W1)
        new_b1 = tf.assign(self.b1, other.b1)
        new_W2 = tf.assign(self.W2, other.W2)
        new_b2 = tf.assign(self.b2, other.b2)
        new_W3 = tf.assign(self.W3, other.W3)
        new_b3 = tf.assign(self.b3, other.b3)

        session.run([new_W1, new_b1, new_W2, new_b2, new_W3, new_b3])
