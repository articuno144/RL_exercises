
# coding: utf-8

# In[41]:


from __future__ import print_function
import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import gym
import matplotlib.pyplot as plt

# initialize variables
n_actions = 2


env = gym.make("CartPole-v0")
observation = env.reset()


input1 = keras.layers.Input(shape=(4,))
input2 = keras.layers.Input(shape=(2,))
concat = keras.layers.Concatenate(axis=-1)([input1,input2])
dense1 = Dense(64, activation='relu')(concat)
out = keras.layers.Dense(1, activation='linear')(dense1)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
model.summary()

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.SGD(lr=0.001))


def Q(s,n_a):
    """takes the state, and the number of actions. Returns a numpy array of estimates of Q(s,a)"""
    q = np.zeros(n_a)
    for i in range(n_a):
        a = np.zeros(n_a)
        a[i] = 1
        q[i] = model.predict([s,a.reshape(1,2)])
    return q


def act(Q):
    """pick action based on Q"""
    epsilon = 0.0001
    if np.sum(np.absolute(Q))< epsilon:
        return np.random.randint(2)
    elif np.random.uniform()<0.7:
        if Q[1]>Q[0]:
            return 1
        else:
            return 0
    else:
        return np.random.randint(2)


# In[51]:


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = - np.ones_like(r)
    for t in reversed(range(0, r.size-1)):
        discounted_r[t] = 0.9*discounted_r[t+1]
    return discounted_r


# In[52]:


states,actions,rewards = [],[],[]
episode_number = 0


# In[55]:
render = False
epi_reward = None
while True:
    # if episode_number%50==49: env.render()
    # 1 iter per frame
    s = observation.reshape(1,4)
    # choose action, apply, get measurements
    q = Q(s,n_actions)
    a = act(q)
    observation, reward, done, info = env.step(a)
    states.append(s)
    action = np.zeros(n_actions)
    action[a] = 1
    actions.append(action)
    rewards.append(reward)
    if done:
        episode_number += 1
        print(np.sum(rewards))
        eps = np.vstack(states)
        epa = np.vstack(actions)
        epr = discount_rewards(np.vstack(rewards))
        # print(epr)
        states,actions,rewards = [],[],[]
        model.fit(x=[eps,epa],y=epr,epochs=1,verbose=0)
        if episode_number%100==0:
        	model.save("model.h5")
        observation = env.reset()