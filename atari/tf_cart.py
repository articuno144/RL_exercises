import tensorflow as tf
import numpy as np
import gym
import time

# model params
input_shape = [4]
output_shape = [2]
hidden_units = 10
learning_rate = 1e-2
# training params
LOAD = True
SAVE = True
VERBOSE = True
GAME = "CartPole-v0"
GAMMA = 0.95
RENDER = False
BATCH_EPISODE = 1


class PolicyGradient:
    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.Session()
        self.s = tf.placeholder(tf.float32, shape=[None]+input_shape)
        self.a = tf.placeholder(tf.int32, shape=[None,])
        self.a_onehot = tf.one_hot(self.a, output_shape[0])
        self.r = tf.placeholder(tf.float32, shape=[None,])
        self.logits, self.probs = self.network(self.s, reuse=False)
        self.record = {'s': [], 'a': []}
        nlogprobs = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.a_onehot, logits=self.logits)
        self.loss = tf.reduce_mean(nlogprobs * self.r)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def network(self, x, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("mlp", reuse=reuse):
            x = tf.layers.flatten(self.s)
            x = tf.layers.dense(
                x, hidden_units, activation='relu') 
            x = tf.layers.dense(
                x, output_shape[0], activation='linear')  # logits
            probs = tf.nn.softmax(x)
            return x, probs

    def run(self, states):
        self.record['s'].append(states[0])
        probs = self.sess.run(self.probs, feed_dict={self.s: states})
        a = np.random.choice(range(probs.shape[1]), size=None, p=probs.ravel())
        self.record['a'].append(a)
        return a

    def train(self):
        loss, _ = self.sess.run([ self.loss, self.train_op], feed_dict={
                      self.s: self.record['s'], self.a: self.record['a'], self.r: self.record['r']})
        # print(loss)
        self.record['s'].clear()
        self.record['a'].clear()

    def load(self, game):
        pass

    def save(self, game):
        pass



def discount_rewards(r):
    """
    take 1D float array of rewards and compute discounted reward 
    from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(len(r))):
            running_add = running_add * GAMMA + r[t]
            discounted_r[t] = running_add

    return (discounted_r - np.mean(discounted_r)) / np.std(discounted_r)


def main():
    sess = tf.Session()
    with sess.as_default():
        model = PolicyGradient()
    if LOAD:
        model.load(GAME)
    env = gym.make(GAME)
    env = env.unwrapped
    env.seed(1)
    writer = tf.summary.FileWriter("logdir", sess.graph)
    score_ph = tf.placeholder(tf.int16)
    tf.summary.scalar("score", score_ph)
    for var in tf.global_variables():
        tf.summary.histogram(var.name, var)
    merged = tf.summary.merge_all()
    tf.get_default_graph().finalize()
    s = env.reset()
    rewards = []
    episode_number = 0
    positive_score = 0
    render = RENDER
    while True:
        if render or episode_number % 50 == 49:
            env.render()
        s = s.reshape([1]+input_shape)
        a = model.run(s)
        s, reward, done, _ = env.step(a)
        rewards.append(reward)
        if reward == 1:
            positive_score += 1
        if done:
            episode_number += 1
            # if np.sum(rewards) > 16:
            #     render = True
            if episode_number % BATCH_EPISODE == 0:
                epr = discount_rewards(np.float32(rewards))
                model.record['r'] = epr
                print("episode {} finished, total positive score: {}".format(
                    episode_number, positive_score))
                summary = sess.run(merged, feed_dict={
                                   score_ph: positive_score})
                writer.add_summary(summary, episode_number)
                positive_score = 0
                # print(model.record)
                model.train()
                if SAVE:
                    model.save(GAME)
            rewards.clear()
            s = env.reset()


if __name__ == "__main__":
    main()
