import tensorflow as tf


class ValueEstimator(object):
    def __init__(self, discount_factor):
        self.discount_factor = discount_factor
        self._build_model()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [1, 4], name="state")
        self.target_value = tf.placeholder(tf.float32, [1, 1], name="target_value")

        self.W1 = tf.Variable(tf.truncated_normal([4, 25], stddev=0.1), name="W1")
        self.b1 = tf.Variable(tf.zeros([25]), name="b1")
        self.hidden = tf.nn.relu(tf.matmul(self.state, self.W1) + self.b1)

        self.W2 = tf.Variable(tf.truncated_normal([25, 1], stddev=0.1), name="W2")
        self.b2 = tf.Variable(tf.zeros([1]), name="b2")
        self.value = tf.matmul(self.hidden, self.W2) + self.b2

        self.loss = tf.squared_difference(self.value, self.target_value)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def td_error(self, session, state, reward, next_state):
        value_for_state = session.run(self.value, feed_dict={self.state: [state]})[0][0]
        value_for_next_state = session.run(self.value, feed_dict={self.state: [next_state]})[0][0]
        td_target = reward + self.discount_factor * value_for_next_state
        td_error = td_target - value_for_state

        return td_error

    def update(self, session, state, reward, next_state, done):
        target_value = reward
        if not done:
            value_for_next_state = session.run(self.value, feed_dict={self.state: [next_state]})[0][0]
            target_value += self.discount_factor * value_for_next_state

        session.run(self.train_op, feed_dict={self.state: [state], self.target_value: [[target_value]]})

    def update_done(self, session, state, reward):
        session.run(self.train_op, feed_dict={self.state: [state], self.target_value: [[reward]]})
