import dataset as ds
import model as md
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPool2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.applications import InceptionV3
from pprint import PrettyPrinter as pp



epochs = 30
batch_size = 1
split_valid=0.2

name, path = md.latestModel()
load_model = path

modelpath = './models/'
modelname = ds.standardModelName(name)
datasetNames = ds.recordingList()
datasetName = datasetNames[0]
#ds.recordingToDataset_allCenter(datasetName)
model_name = 'BN_INCEP' #'InceptionV3_features'

gen_train, gen_valid, info = ds.loadDatasetGenerators(datasetName, batch_size=batch_size)


#pp.pprint(info)
features = InceptionV3(include_top=False,input_shape=info['input_shape'], pooling='max')
x = Dense(128, activation="relu")(features.output)
x = Dense(1)(x)
model = Model(features.input, x, name="test")

'''
if load_model is not None:
    print('Loading weights', load_model)
    model.load_weights(load_model)
''';


filepath= modelpath + modelname + "-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
model.fit_generator(gen_train, info['n_train_batch'], epochs=epochs, validation_data=gen_valid, validation_steps=info['n_valid_batch'],callbacks=callbacks_list)

#model.save(modelpath+modelname+'.h5')