import numpy as np
import tensorflow as tf


class Estimator(object):
    def __init__(self, update_discount_factor, name):
        self.discount_factor = update_discount_factor
        self._build_model(name)

    def _build_model(self, name):
        with tf.name_scope(name):
            self.input = tf.placeholder(tf.float32, [None, 84, 84, 4], name="input")
            self.target_output = tf.placeholder(tf.float32, [None, 4], name="target_output")
            self.dropout_keep_probability = tf.placeholder(tf.float32)

            with tf.name_scope("cnn"):
                self.conv1 = tf.layers.conv2d(self.input, filters=32, kernel_size=[8, 8], strides=(4, 4), padding="valid", activation=tf.nn.relu)
                self.conv2 = tf.layers.conv2d(self.conv1, filters=64, kernel_size=[4, 4], strides=(2, 2), padding="valid", activation=tf.nn.relu)
                self.conv3 = tf.layers.conv2d(self.conv2, filters=64, kernel_size=[3, 3], strides=(1, 1), padding="valid", activation=tf.nn.relu)
                self.conv3_flat = tf.reshape(self.conv3, shape=[-1, 7*7*64])
                self.dense = tf.layers.dense(self.conv3_flat, units=512, activation=tf.nn.relu)
                self.dense_dropout = tf.nn.dropout(self.dense, self.dropout_keep_probability)
                self.output = tf.layers.dense(self.dense_dropout, units=4)

            with tf.name_scope("predictions"):
                self.predictions = tf.argmax(self.output, 1, name="predictions")

            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.squared_difference(self.output, self.target_output))

                # regularization
                # for w in [self.W1, self.W2, self.W3]:
                #     self.loss += 0.001 * tf.reduce_sum(tf.square(w))

            with tf.name_scope("train"):
                self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def predict(self, session, states_batch):
        return session.run(self.predictions, feed_dict={self.input: states_batch, self.dropout_keep_probability: 1.0})

    def td_errors(self, session, second_estimator, states_batch, actions_batch, rewards_batch, next_states_batch):
        batch_size = len(states_batch)
        next_best_actions = session.run(self.predictions, feed_dict={self.input: next_states_batch, self.dropout_keep_probability: 1.0})

        next_q_values = session.run(second_estimator.output, feed_dict={second_estimator.input: next_states_batch, second_estimator.dropout_keep_probability: 1.0})

        target_q_values = session.run(self.output, feed_dict={self.input: states_batch, self.dropout_keep_probability: 1.0})

        errors = np.zeros(batch_size)
        for i in range(batch_size):
            errors[i] = abs(target_q_values[i, actions_batch[i]] - rewards_batch[i] - self.discount_factor * next_q_values[i, next_best_actions[i]])

        return errors

    def update(self, session, second_estimator, states_batch, actions_batch, rewards_batch, next_states_batch, done_batch):
        batch_size = states_batch.shape[0]
        next_best_actions = session.run(self.predictions, feed_dict={self.input: next_states_batch, self.dropout_keep_probability: 1.0})

        next_q_values = session.run(second_estimator.output, feed_dict={second_estimator.input: next_states_batch, second_estimator.dropout_keep_probability: 1.0})

        target_q_values = session.run(self.output, feed_dict={self.input: states_batch, self.dropout_keep_probability: 1.0})

        for i in range(batch_size):
            if done_batch[i]:
                target_q_values[i, actions_batch[i]] = rewards_batch[i]
            else:
                target_q_values[i, actions_batch[i]] = rewards_batch[i] + self.discount_factor * next_q_values[i, next_best_actions[i]]

        session.run(self.train_step, feed_dict={self.input: states_batch, self.target_output: target_q_values, self.dropout_keep_probability: 0.75})
