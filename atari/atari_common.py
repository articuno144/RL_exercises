import gym
import numpy as np
import tensorflow as tf


def train(game, reward_func, model=None, input_shape=None, n_actions=None, verbose=False, save=True, load=True, render=False, prepro=None, batch_episode=10, epoch_per_batch=1):
    if prepro is not None:
        assert input_shape is not None, "if you specify a preprocess function, you must also specify the input shape after preprocessing"
    model = model(input_shape, n_actions, verbose=verbose)
    if load:
        tf.reset_default_graph()
        model.load(game)
    env = gym.make(game)
    if input_shape is None:
        input_shape = list(env.observation_space.shape)
    if n_actions is None:
        n_actions = env.action_space.n

    sess = tf.Session()
    writer = tf.summary.FileWriter("logdir", sess.graph)
    indicator_ph = tf.placeholder(tf.int16)
    tf.summary.scalar("indicator", indicator_ph)
    merged = tf.summary.merge_all()

    observation = env.reset()
    states, actions, rewards = [], [], []
    episode_number = 0
    indicator = 0
    render = render
    history = []
    epi_reward = None
    prev_s = None
    while True:
        if render or episode_number % 50 == 49:
            env.render()
        if prepro is not None:
            cur_s = prepro(observation).reshape(input_shape)
        s = cur_s - prev_s if prev_s is not None else np.zeros(input_shape)
        prev_s = cur_s
        s = s.reshape([1]+input_shape)
        q = model.Q(s)
        a = model.act(q)
        observation, reward, done, _ = env.step(a+2)
        states.append(s)
        action = np.zeros(n_actions)
        action[a] = 1
        actions.append(action)
        rewards.append(reward)
        if reward == 1:
            indicator += 1
        if done:
            episode_number += 1
            if epi_reward == None:
                epi_reward = np.sum(rewards)
                prev_sum = epi_reward
            else:
                epi_reward = np.sum(rewards)-prev_sum
                prev_sum = np.sum(rewards)
            history.append(epi_reward)
            if np.sum(rewards) > 16:
                render = True
            if episode_number % batch_episode == 0:
                eps = np.vstack(states)
                epa = np.vstack(actions)
                epr = reward_func(np.vstack(rewards))
                states, actions, rewards = [], [], []
                epi_reward = None
                print("episode {} finished, total positive score: {}".format(
                    episode_number, indicator))
                model.model.fit(x=[eps, epa], y=epr,
                                epochs=epoch_per_batch, verbose=verbose)
                summary = sess.run(merged, feed_dict={indicator_ph: indicator})
                writer.add_summary(summary, episode_number)
                indicator = 0
                if save:
                    model.save(game)
            observation = env.reset()
