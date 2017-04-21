import dataset as ds
import model as md
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

epochs = 30
batch_size = 1
split_valid=0.2

datasetName = 'lake_lap_final' # jungle, lake, try, lake_lap_clean, lake_lap_final
modelname = ds.standardModelName(datasetName+'_nvidia')

## Prepare dataset
# force=True rebuilds the dataset
# reindex_only=True rebuild only the index file (ex: change steer correction formula, paths, etc)
ds.recordingToDataset_allCenter(datasetName) # parse a recording into a dataset (if not done yet)

#load dataset generator and metrics
gen_train, gen_valid, info = ds.loadDatasetGenerators(datasetName, batch_size=batch_size )
# print(info)

# create the model a eventually preload the weights (set to None or remove to disable)
load_weight = md.models_path + 'lake_lap_final_nvidia_20170420-162101-00-0.00416.h5'
model = md.nvidia_driving_team(input_shape=info['input_shape'], load_weight=load_weight)

# Intermediate model filename template
filepath= md.models_path + modelname + "-{epoch:02d}-{val_loss:.5f}.h5"
# save model after every epoc, only if improved the val_loss.
# very handy (with a proper environment (2 GPUs anywhere) you can test your model while it still train)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
# detect stop in gain on val_loss between epocs and terminate early, avoiding unnecessary computation cycles.
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1)
callbacks_list = [checkpoint, earlystopping]

model.compile(loss="mse", optimizer="sgd")
history_object = model.fit_generator(gen_train, info['n_train_batch'],verbose=1, epochs=epochs, validation_data=gen_valid, validation_steps=info['n_valid_batch'], callbacks=callbacks_list)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

