#%%
from pyJoules.energy_meter import measure_energy

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten

data = tf.keras.datasets.mnist.load_data(path="mnist.npz")

X_train,y_train,X_test,y_test = data[0][0],data[0][1],data[1][0],data[1][1]
X_train,X_test = np.expand_dims(X_train, axis=-1),np.expand_dims(X_test, axis=-1)
y_train,y_test = tf.keras.utils.to_categorical(y_train,10),tf.keras.utils.to_categorical(y_test,10)

#%%
model = tf.keras.models.Sequential()
model.add(Conv2D(40, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(10,activation = 'softmax'))

model.compile(loss='CategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])

#%%


@measure_energy
def foo():
	# Instructions to be evaluated.
    model.fit(X_train,y_train,epochs=5)


foo()
# %%
