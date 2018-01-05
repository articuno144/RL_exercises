
# coding: utf-8

# In[41]:


from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import gym
import matplotlib.pyplot as plt

# initialize variables
n_actions = 2


env = gym.make("SpaceInvaders-v0")
observation = env.reset()
prev_s = None

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I < 50] = 0 # erase background (background type 1)
    #I[I > 100] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I


img = keras.layers.Input(shape=(80,80,1))
conv1 = Conv2D(8, (8, 8), activation='relu')(img)
maxpool1 = MaxPooling2D(pool_size=(4, 4))(conv1)
conv2 = Conv2D(8, (5, 5), activation='relu')(maxpool1)
maxpool2 = MaxPooling2D(pool_size=(4, 4))(conv2)
flatten = Flatten()(maxpool2)
input2 = keras.layers.Input(shape=(n_actions,))
concat = keras.layers.Concatenate(axis=-1)([input2,flatten])
dense1 = Dense(64, activation='relu')(concat)
dense2 = Dense(64, activation='relu')(dense1)
out = keras.layers.Dense(1, activation='linear')(dense2)
model = keras.models.Model(inputs=[img, input2], outputs=out)
model.summary()

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam())


def Q(s,n_a):
    """takes the state, and the number of actions. Returns a numpy array of estimates of Q(s,a)"""
    q = np.zeros(n_a)
    for i in range(n_a):
        a = np.zeros(n_a)
        a[i] = 1
        q[i] = model.predict([s,a.reshape(1,2)])
    return q



def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))



def act(Q):
    """pick action based on Q"""
    epsilon = 0.0001
    if np.sum(np.absolute(Q))< epsilon:
        return np.random.randint(2)
    elif np.random.uniform()<0.85:
        #tendency = (Q[1] - Q[0])*50/np.sum(np.abs(Q))
        #if np.random.uniform()<sigmoid(tendency):
        if Q[1] > Q[0]:
        	return 1
        else:
        	return 0
    else:
        return np.random.randint(2)


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * 0.98 + r[t]
        discounted_r[t] = running_add
    return discounted_r


states,actions,rewards = [],[],[]
episode_number = 0


render = True
history = []
plt.ion()
plt.hold(False)
epi_reward = None
while True:
    if render or episode_number%50==49: env.render()
    # 1 iter per frame
    cur_s = prepro(observation).reshape([80,80,1])
    s = cur_s - prev_s if prev_s is not None else np.zeros([80,80,1])
    prev_s = cur_s
    s = s.reshape([1,80,80,1])
    # choose action, apply, get measurements
    q = Q(s,n_actions)
    a = act(q)
    observation, reward, done, info = env.step(a+2)
    states.append(s)
    action = np.zeros(n_actions)
    action[a] = 1
    actions.append(action)
    rewards.append(reward)
    if reward == 1: print("!!!!!")
    # when finished, calculate rewards, train Q against these rewards
    if done:
        episode_number += 1
        if epi_reward == None:
        	epi_reward = np.sum(rewards)
        	prev_sum = epi_reward
        else:
        	epi_reward = np.sum(rewards)-prev_sum
        	prev_sum = np.sum(rewards)
        history.append(epi_reward)
        if np.sum(rewards)>10:
        	render = True
        if episode_number%10==0:
	        eps = np.vstack(states)
	        epa = np.vstack(actions)
	        epr = discount_rewards(np.vstack(rewards))
	        states,actions,rewards = [],[],[]
	        epi_reward = None
        	print("episode "+ str(episode_number) + " finished")
        	model.fit(x=[eps,epa],y=epr,epochs=1,verbose=1)
        	plt.plot(history)
        	plt.pause(0.0001)
        	plt.show()
        	plt.pause(0.0001)
        if episode_number%100==0:
        	model.save("model4.h5")
        observation = env.reset()
        prev_s = None