from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPool2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.applications import InceptionV3
import glob, os

models_path = './models/'

def latestModel():
    max = 0
    latest_path = None
    model_files = glob.glob(models_path+'*.h5')
    for model_file in model_files:
        time = os.path.getmtime(model_file)
        if time > max:
            latest_path = model_file
            max = time
    filename = latest_path.split("/")[-1]
    model_name = filename.split("-")[0]
    return model_name, latest_path

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

def InceptionV3_retrain(input_shape, name=None):
    features = InceptionV3(include_top=False,input_shape=input_shape, pooling='max')
    x = Dense(128, activation="relu")(features.output)
    x = Dense(1)(x)
    model = Model(features.input, x, name=name)
    return model

def InceptionV3_bottlenecks(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape) )
    model.add(Dense(512))
    model.add(Dense(256))
    model.add(Dense(1))
    return model