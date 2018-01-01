
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# In[2]:


import gym


# In[3]:


# initialize variables
n_actions = 2


# In[4]:


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None


# In[ ]:


img = keras.layers.Input(shape=(80,80,1))
conv1 = Conv2D(16, (3, 3), activation='relu')(img)
maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, (3, 3), activation='relu')(maxpool1)
maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(16, (3, 3), activation='relu')(maxpool2)
maxpool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
flatten = Flatten()(maxpool3)
input2 = keras.layers.Input(shape=(n_actions,))
concat = keras.layers.Concatenate(axis=-1)([input2,flatten])
dense1 = Dense(64, activation='relu')(concat)
dense2 = Dense(32, activation='relu')(dense1)
out = keras.layers.Dense(1, activation='relu')(dense2)
model = keras.models.Model(inputs=[img, input2], outputs=out)
model.summary()

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Adam())


# In[ ]:


s = np.zeros([1,80,80,1])
a = np.zeros([1,2])
print(model.predict([s,a]))


# In[ ]:


def Q(s,n_a):
    """takes the state, and the number of actions. Returns a numpy array of estimates of Q(s,a)"""
    q = np.zeros(n_a)
    for i in range(n_a):
        a = np.zeros(n_a)
        a[i] = 1
        q[i] = model.predict([s,a])


# In[ ]:
