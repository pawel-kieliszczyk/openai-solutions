import tensorflow as tf


class PolicyEstimator(object):
    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.state = tf.placeholder(tf.float32, [1, 4], name="state")
        self.action = tf.placeholder(tf.int32, [1], name="action")
        self.advantage = tf.placeholder(tf.float32, [1], name="advantage")

        self.W = tf.Variable(tf.truncated_normal([4, 2], stddev=2.0), name="W")
        self.b = tf.Variable(tf.zeros([2]), name="b")
        self.action_probs = tf.nn.softmax(tf.matmul(self.state, self.W) + self.b)

        self.picked_action_prob = tf.gather(self.action_probs[0], self.action)
        self.loss = -tf.log(self.picked_action_prob) * self.advantage

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def predict(self, session, state):
        return session.run(self.action_probs, feed_dict={self.state: [state]})[0]

    def update(self, session, state, advantage, action):
        feed_dict = {self.state: [state], self.advantage: [advantage], self.action: [action]}
        session.run(self.train_op, feed_dict)
