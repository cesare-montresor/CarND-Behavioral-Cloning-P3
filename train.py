import dataset as ds
import models
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPool2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.applications import InceptionV3

batch_size = 32
split_valid=0.2
image_shape=(160,320,3)

modelpath = './models/'
modelname = 'first'
datasetNames = ds.recordingList()
datasetName = datasetNames[0]
#ds.recordingToDataset_allCenter(datasetName)
model_name = 'InceptionV3_features'

ds.datasetToBottleneck_InceptionV3(datasetName,image_shape, limit=2, model_name = 'InceptionV3_features', reindex_only=False)
gen_train, gen_valid, info = ds.loadBottleneckGenerators(datasetName, model_name, batch_size=batch_size)
print(info)

model = Sequential()
model.add(Dense(1024, input_shape=info['input_shape']))
model.add(Dense(512))
model.add(Dense(1))

filepath= modelpath + modelname + "-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])
model.fit_generator(gen_train, info['n_train_batch'], epochs=5, validation_data=gen_valid, validation_steps=info['n_valid_batch'],callbacks=callbacks_list)

#model.save(modelpath+modelname+'.h5')