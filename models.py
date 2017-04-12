from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPool2D, Cropping2D
from keras.callbacks import ModelCheckpoint

def LeNet(image_shape):
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=image_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Conv2D(6, 5, 5, border_mode="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, 5, 5, border_mode="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation="relu"))
    model.add(Dense(84, activation="relu"))
    model.add(Dense(1))

