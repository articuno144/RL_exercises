import numpy as np
import keras
import os
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


DESPERATION_R = -0.8


def softmax(arr):
    exp_arr = np.exp(arr)
    return exp_arr / np.sum(exp_arr)


class Model:
    def __init__(self, input_size, n_actions, verbose=True):
        """
        Initializes the model.
          Args:
            input_size: image size provided by the environment in HWC, e.g.[28,28,1]
            n_actions: number of actions that the agent can choose from, e.g. 2
            verbose: print model summary
            load: whether to load the model
        """
        self.input_size = input_size
        self.n_actions = n_actions
        self.model = self.network()
        self.model.compile(loss='mean_squared_error',
                           optimizer=keras.optimizers.Adam(lr=0.0001))
        if verbose:
            self.model.summary()

    def network(self):
        w, h, c = self.input_size
        img = keras.layers.Input(shape=(w, h, c))
        layer = self.conv(img)
        layer = self.conv(layer)
        layer = self.conv(layer)
        flatten = Flatten()(layer)
        input2 = keras.layers.Input(shape=(self.n_actions,))
        concat = keras.layers.Concatenate(axis=-1)([input2, flatten])
        dense1 = Dense(64, activation='relu')(concat)
        dense2 = Dense(64, activation='relu')(dense1)
        out = keras.layers.Dense(1, activation='linear')(dense2)
        return keras.models.Model(inputs=[img, input2], outputs=out)

    def conv(self, input_layer):
        return MaxPooling2D(pool_size=(2, 2))(Conv2D(8, (3, 3), activation='relu')(input_layer))

    def Q(self, s):
        """takes the state, and the number of actions. Returns a numpy array of estimates of Q(s,a)"""
        q = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            a = np.zeros(self.n_actions)
            a[i] = 1
            q[i] = self.model.predict([s, a.reshape(1, self.n_actions)])
        return q

    def act(self, Q):
        """pick action based on Q"""
        if np.max(Q) < DESPERATION_R:
            return np.random.randint(self.n_actions)
        # return list(np.random.uniform() > np.cumsum(softmax(Q))).index(False)
        return 0 if Q[0] > Q[1] else 1

    def save(self, game, dir="save.h5"):
        assert type(game) is str, "The name of the game should be a string"
        if not os.path.exists("save"):
            os.mkdir("save")
        if not os.path.exists("save/"+game):
            os.mkdir("save/"+game)
        self.model.save("save/"+game+"/"+dir)

    def load(self, game, dir="save.h5"):
        del self.model
        self.model = load_model("save/"+game+"/"+dir)


if __name__ == "__main__":
    model = Model([80, 80, 1], 2)
