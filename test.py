import dataset as ds
import model as md
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPool2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.applications import InceptionV3
import numpy as np

input_shape = (1,)
features = InceptionV3(include_top=False,input_shape=input_shape, pooling='max')
x = Dense(512, activation="relu")(features.output)
x = Dense(512, activation="relu")
x = Dense(1, activation="relu")(x)
model = Model(features.input, x, name="test")


model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
model.fit(x=X,y=y,shuffle=True,epochs=30,validation_split=0.2)



