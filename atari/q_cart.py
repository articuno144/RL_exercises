import gym
import tensorflow as tf
import numpy as np

input_dim = 4
output_dim = 2
hidden_dim = 16
fail_score = -10
goal = 1000


class Q:
    def __init__(self, input_dim, output_dim, hidden_dim, lr=1e-2, gamma=0.9):
        # high learning rate, batch learning, could potentially unlearn things
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        x = self.x = tf.placeholder(tf.float32, shape=[None, input_dim])
        self.r = tf.placeholder(tf.float32, shape=[None])
        q = self.q = self.net(x)  # [None, output_dim]
        self.a = tf.argmax(q,1)
        self.q_a = tf.reduce_max(q,1)
        self.s_history = []
        loss = tf.losses.mean_squared_error(self.r,self.q_a)
        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()

                
    def net(self, x):
        x = tf.layers.dense(x, self.hidden_dim, 'relu')
        x = tf.layers.dense(x, self.output_dim, 'linear')
        return x

    def act(self, s):
        self.s_history.append(s)
        s = s.reshape([1, self.input_dim])
        return self.sess.run(self.a, feed_dict={self.x:s})[0]

    def train(self, r):
        r = r[:-1]
        q_a_next = self.sess.run(self.q_a, feed_dict={self.x:self.s_history[1:]})
        r += self.gamma * q_a_next
        self.sess.run(self.train_op,feed_dict={self.x:self.s_history[:-1], self.r:r})
        self.s_history.clear()


def main():
    env = gym.make("CartPole-v0")
    env = env.unwrapped
    q = Q(input_dim, output_dim, hidden_dim)
    s = env.reset()
    episode_number = 0
    positive_score = 0
    r = []
    while(True):
        a = q.act(s)
        s, reward, done, _ = env.step(a)
        r.append(reward)
        if reward == 1:
            positive_score += 1
        if done:
            episode_number += 1
            print("episode {}, positive score {}".format(episode_number, positive_score))
            if positive_score > goal:
                break
            env.reset()
            positive_score = 0
            r[-2] = fail_score  # for simplicity, penalize r instead of q
            q.train(np.array(r))
            r.clear()
    print("Success!")

if __name__ == "__main__":
    main()