import tensorflow as tf


class ValueEstimator(object):
    def __init__(self, discount_factor):
        self.discount_factor = discount_factor
        self._build_model()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [1, 4], name="state")
        self.target_predictions = tf.placeholder(tf.float32, [1, 2], name="target_predictions")

        self.W1 = tf.Variable(tf.truncated_normal([4, 25], stddev=0.1), name="W1")
        self.b1 = tf.Variable(tf.zeros([25]), name="b1")
        self.hidden = tf.nn.relu(tf.matmul(self.state, self.W1) + self.b1)

        self.W2 = tf.Variable(tf.truncated_normal([25, 2], stddev=0.1), name="W2")
        self.b2 = tf.Variable(tf.zeros([2]), name="b2")
        self.predictions = tf.matmul(self.hidden, self.W2) + self.b2

        self.loss = tf.squared_difference(self.predictions, self.target_predictions)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def predict(self, session, state):
        return session.run(self.predictions, feed_dict={self.state: [state]})[0]

    def update(self, session, state, action, reward, next_state, next_action):
        predictions_state = session.run(self.predictions, feed_dict={self.state: [state]})[0]
        predictions_next_state = session.run(self.predictions, feed_dict={self.state: [next_state]})[0]

        predictions_state[action] = reward + self.discount_factor * predictions_next_state[next_action]

        session.run(self.train_op, feed_dict={self.state: [state], self.target_predictions: [predictions_state]})

    def update_done(self, session, state, action, reward):
        predictions_state = session.run(self.predictions, feed_dict={self.state: [state]})[0]

        predictions_state[action] = reward

        session.run(self.train_op, feed_dict={self.state: [state], self.target_predictions: [predictions_state]})
