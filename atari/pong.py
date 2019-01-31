import gym
import numpy as np
from model import Model
from atari_common import train

LOAD = True
SAVE = True
VERBOSE = True
GAME = "Pong-v0"
GAMMA = 0.98
RENDER = False


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
    return discounted_r


if __name__ == "__main__":
    train(GAME, discount_rewards, Model, input_shape=[
          80, 80, 1], n_actions=2, verbose=VERBOSE, save=SAVE, load=LOAD, render=RENDER, prepro=prepro, batch_episode=10, epoch_per_batch=1)
