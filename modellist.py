from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPool2D, Cropping2D, Dropout
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


def nvidia_driving_team(input_shape,name="nvidia_v1",load_weight=None):
    model = Sequential()
    model.add(Cropping2D( cropping=((70,25),(0,0)),input_shape=input_shape ))
    model.add(Lambda(lambda x: (x / 255) - 0.5))  # normalization layer
    model.add(Conv2D(24, kernel_size=(5, 5), activation="relu", strides=(2,2) ))
    model.add(Conv2D(48, kernel_size=(5, 5), activation="relu", strides=(2,2) ))
    model.add(Conv2D(72, kernel_size=(5, 5), activation="relu", strides=(2,2) ))
    model.add(Conv2D(96, kernel_size=(3, 3), activation="relu") )
    model.add(Conv2D(120, kernel_size=(3, 3), activation="relu") )
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    model.name = name

    if load_weight is not None and os.path.isfile(load_weight):
        print('Loading weights', load_weight)
        model.load_weights(load_weight)
    else:
        print('Loading weights failed', load_weight)

    return model




#### OTHER MODELS TESTED ####

def LeNet(input_shape):
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Conv2D(6, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation="relu"))
    model.add(Dense(84, activation="relu"))
    model.add(Dense(1))
    return model

def InceptionV3_retrain(input_shape, name='incept_retrain', load_weight=None):
    features = InceptionV3(include_top=False,input_shape=input_shape, pooling='max')
    x = Dense(128, activation="relu")(features.output)
    x = Dense(1)(x)
    model = Model(features.input, x, name=name)
    if load_weight is not None and os.path.isfile(load_weight):
        print('Loading weights', load_weight)
        model.load_weights(load_weight)
    else:
        print('Loading weights failed', load_weight)
    return model

def InceptionV3_bottlenecks(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape) )
    model.add(Dense(512))
    model.add(Dense(256))
    model.add(Dense(1))
    return model



def model_P2(input_shape,name="old_friend_v1",load_weight=None):
    model = Sequential()
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255) - 0.5))  # normalization layer
    model.add(Conv2D(10, kernel_size=(3, 3), activation="relu" ))
    model.add(MaxPool2D())
    model.add(Conv2D(20, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPool2D())
    model.add(Conv2D(40, kernel_size=(3, 3), activation="relu", padding="VALID"))
    model.add(Conv2D(60, kernel_size=(3, 3), activation="relu", padding="VALID"))
    model.add(MaxPool2D())
    model.add(Conv2D(80, kernel_size=(3, 3), activation="relu", padding="VALID"))
    model.add(Flatten())
    model.add(Dense(600))
    model.add(Dropout(0.6))
    model.add(Dense(400))
    model.add(Dropout(0.6))
    model.add(Dense(200))
    model.add(Dropout(0.6))
    model.add(Dense(100))
    model.add(Dropout(0.6))
    model.add(Dense(1))
    model.name = name

    return model