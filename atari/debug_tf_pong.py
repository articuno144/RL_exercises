import tensorflow as tf
import numpy as np
import gym
import time
import os

# model params
input_shape = [80, 80, 1]
output_shape = [2]
hidden_units = 128
learning_rate = 1e-2
# training params
LOAD = False
SAVE = True
VERBOSE = True
GAME = "Pong-v0"
GAMMA = 0.98
RENDER = False
BATCH_EPISODE = 1


class PolicyGradient:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None]+input_shape)
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.Session()
        self.global_step = tf.Variable(
            0, trainable=False, name="global_step")
        self.prob_grad, self.a = self.network(reuse=False)
        self.prob_grads = []
        self.prob_as = []
        self.grad = [tf.placeholder(tf.float32, shape=var.shape)
                     for var in self.train_vars]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update = self.optimizer.apply_gradients(
            zip(self.grad, self.train_vars), global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables())
        self.sess.run(tf.global_variables_initializer())

    def network(self, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("mlp", reuse=reuse):
            x = tf.layers.flatten(self.x)
            x = tf.layers.dense(x, hidden_units, activation='relu')
            self.logits = tf.layers.dense(x, output_shape[0], activation='linear')
            x = tf.nn.log_softmax(self.logits)
            a = tf.reshape(tf.cast(tf.multinomial(
                x, 1), tf.int32), [-1])  # (batch,)
            self.prob_a = tf.gather_nd(x, tf.transpose([
                tf.range(tf.shape(x)[0]), a]))  # (batch,)
            self.train_vars = tf.trainable_variables()
            prob_grad = tf.gradients(self.prob_a, self.train_vars)
            self.actual_proba = tf.exp(self.prob_a)
            return prob_grad, a

    def run(self, state, reuse=tf.AUTO_REUSE):
        p, logits, prob_grad, a, prob_a = self.sess.run(
            [self.actual_proba, self.logits, self.prob_grad, self.a, self.prob_a], feed_dict={self.x: state})
        print(p, logits, prob_a, a)
        self.prob_grads.append(prob_grad)
        self.prob_as.append(prob_a)
        return a

    def backprop(self, reward_hist):
        grad = [np.zeros(var.shape, dtype=np.float32)
                for var in self.train_vars]
        for i in range(len(reward_hist)):
            for j in range(len(self.train_vars)):
                grad[j] += - self.prob_grads[i][j]*reward_hist[i]
        for j in range(len(self.train_vars)):
            grad[j] /= len(reward_hist)
        self.sess.run(self.update, feed_dict={
                      ph: g for ph, g in zip(self.grad, grad)})

    def clear(self):
        self.prob_grads.clear()

    def load(self, game):
        try:
            self.saver.restore(self.sess, "save/Pong-v0/tf/tf_pg_model.ckpt")
        except:
            print("save file not found, proceeding with default initialization")

    def save(self, game):
        self.saver.save(self.sess, "save/Pong-v0/tf/tf_pg_model.ckpt")


def prepro(I):
    """ 
    prepro 210x160x3 uint8 frame into (80x80)
    from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I


def discount_rewards(r):
    """
    take 1D float array of rewards and compute discounted reward 
    from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return (discounted_r - np.mean(discounted_r))/np.std(discounted_r)


def main():
    sess = tf.Session()
    with sess.as_default():
        model = PolicyGradient()
    if LOAD:
        model.load(GAME)
    else:
        for f in os.listdir("logdir/tf_policy_gradient/"):
            os.remove("logdir/tf_policy_gradient/"+f)
    env = gym.make(GAME)
    writer = tf.summary.FileWriter("logdir/tf_policy_gradient", sess.graph)
    score_ph = tf.placeholder(tf.int16)
    epr_ph = tf.placeholder(tf.float32)
    prob_a_ph = tf.placeholder(tf.float32)
    tf.summary.scalar("score", score_ph)
    tf.summary.histogram("discounted rewards", epr_ph)
    tf.summary.histogram("action prob", prob_a_ph)
    for var in model.train_vars:
        tf.summary.histogram(var.name, var)
    merged = tf.summary.merge_all()
    observation = env.reset()
    rewards = []
    episode_number = sess.run(model.global_step)
    positive_score = 0
    render = RENDER
    prev_s = None
    while True:
        if render or episode_number % 50 == 49:
            env.render()
        if prepro is not None:
            cur_s = prepro(observation).reshape(input_shape)
        s = cur_s - prev_s if prev_s is not None else np.zeros(input_shape)
        prev_s = cur_s
        s = s.reshape([1]+input_shape)
        a = model.run(s, reuse=True)
        observation, reward, done, _ = env.step(a+2)
        rewards.append(reward)
        if reward == 1:
            positive_score += 1
        if done:
            episode_number += 1
            if np.sum(rewards) > 16:
                render = True
            if episode_number % BATCH_EPISODE == 0:
                epr = discount_rewards(np.float32(np.vstack(rewards)))
                model.backprop(epr)
                print("episode {} finished, total positive score: {}".format(
                    episode_number, positive_score))
                summary = sess.run(merged, feed_dict={
                                   score_ph: positive_score, epr_ph: np.squeeze(epr), prob_a_ph: np.squeeze(np.array(model.prob_as))})
                writer.add_summary(summary, sess.run(model.global_step))
                positive_score = 0
                model.clear()
                rewards.clear()
                if SAVE:
                    model.save(GAME)
            observation = env.reset()


if __name__ == "__main__":
    main()
