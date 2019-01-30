import gym
import numpy as np
from model import Model


LOAD = True
SAVE = True
VERBOSE = True
GAME = "Pong-v0"

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
        running_add = running_add * 0.98 + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():
    env = gym.make(GAME)
    n_actions = 2  # somehow the action space has a default size of 6, will check
    input_shape = [80, 80, 1]  # after prepro
    observation = env.reset()
    model = Model(input_shape, n_actions, verbose=VERBOSE)
    if LOAD:
        model.load(GAME)
    states, actions, rewards = [], [], []
    episode_number = 0
    positive_score = 0
    render = False
    history = []
    epi_reward = None
    prev_s = None
    while True:
        if render or episode_number % 50 == 49:
            env.render()
        cur_s = prepro(observation).reshape([80, 80, 1])
        s = cur_s - prev_s if prev_s is not None else np.zeros([80, 80, 1])
        prev_s = cur_s
        s = s.reshape([1, 80, 80, 1])
        q = model.Q(s)
        a = model.act(q)
        observation, reward, done, _ = env.step(a+2)
        states.append(s)
        action = np.zeros(n_actions)
        action[a] = 1
        actions.append(action)
        rewards.append(reward)
        if reward == 1:
            positive_score += 1
        if done:
            episode_number += 1
            if epi_reward == None:
                epi_reward = np.sum(rewards)
                prev_sum = epi_reward
            else:
                epi_reward = np.sum(rewards)-prev_sum
                prev_sum = np.sum(rewards)
            history.append(epi_reward)
            if np.sum(rewards) > 10:
                render = True
            if episode_number % 10 == 0:
                eps = np.vstack(states)
                epa = np.vstack(actions)
                epr = discount_rewards(np.vstack(rewards))
                states, actions, rewards = [], [], []
                epi_reward = None
                print("episode {} finished, total positive score: {}".format(
                    episode_number, positive_score))
                model.model.fit(x=[eps, epa], y=epr, epochs=1, verbose=VERBOSE)
                positive_score = 0
                if SAVE:
                    model.save(game=GAME)
            observation = env.reset()
            prev_s = None


if __name__ == "__main__":
    main()
