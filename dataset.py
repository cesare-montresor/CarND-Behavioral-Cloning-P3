import csv, cv2, glob, os, math
import numpy as np
import pickle
import numpy as np
from shutil import copyfile
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.applications import InceptionV3
import h5py, pickle
import datetime
import json

cvsname = 'driving_log.csv'
trainfile = 'train.p'
recording_path = './recordings/'
datasets_path = './datasets/'
default_batch_size = 32

def standardModelName(name=None):
    name = name if name is not None else 'model_'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    return name

def datasetToBottleneck_InceptionV3(name, image_shape, batch_size=default_batch_size, force=False, model_name='InceptionV3_features', limit=None,reindex_only=False):
    features = InceptionV3(include_top=False, input_shape=image_shape)
    return datasetToBottleneck(name, features, batch_size=batch_size, limit=limit, model_name=model_name, reindex_only=reindex_only)

def datasetToBottleneck(name, model, model_name=None , batch_size=default_batch_size, force=False, limit=None, reindex_only=False):
    model_name = standardModelName(model_name)
    dst_path = datasets_path + name + '/'
    dataset=loadDataset(name)

    #ensure that the dir exists
    X_one, y_one = dataset[0]
    parts = X_one.split('/')
    parts[-2] = model_name
    os.makedirs("/".join(parts[:-1]), exist_ok=True)

    train_data = []
    n_train = len(dataset) if limit is None else limit
    for offset in range(0, n_train, batch_size):
        offset_end = min(offset + batch_size,n_train)
        dataset_batch = dataset[offset:offset_end]
        X_batch = []
        X_batch_paths = []
        for sample in dataset_batch:
            X_path,y = sample
            parts = X_path.split('/')
            parts[-2] = model_name
            X_path_bn = "/".join(parts) + '.p'
            train_data.append([X_path_bn,y])

            if not reindex_only and (force or not os.path.isfile(X_path_bn)):
                X = cv2.imread(X_path)
                X_batch.append(X)
                X_batch_paths.append(X_path_bn)

        if not reindex_only and len(X_batch)>0:
            X_batch = np.array(X_batch)
            X_batch_bn=model.predict(X_batch)

            for i in range(len(X_batch_bn)):
                X_bn = X_batch_bn[i]
                X_path_bn = X_batch_paths[i]
                with open(X_path_bn, 'wb') as picklefile:
                    pickle.dump(X_bn,picklefile)

    with open(dst_path+model_name+'/'+trainfile, 'wb') as picklefile:
        pickle.dump(train_data, picklefile)
        #print(train_data)
    return model_name



def recordingToDataset_allCenter(name,correction=0.1,force=False,limit=None):  #performs center
    #print('name',name)
    src_path = recording_path + name + '/'
    dst_path = datasets_path + name + '/'
    train_path = dst_path+trainfile

    os.makedirs(dst_path+'IMG/', exist_ok=True)

    train_data = []

    if os.path.isfile(train_path) and not force:
        return  #avoid to repeat the import

    cnt = 0
    with open(src_path+cvsname) as csvfile:
        csvreader = csv.reader(csvfile)
        for line in csvreader:
            center_path, left_path, right_path, steering, throttle, brake, speed = line

            center_path = center_path.strip()
            left_path = left_path.split("|")
            left_path = list(map(lambda x: x.strip(), left_path))     # left_path = left_path.strip()
            right_path = right_path.split("|")
            right_path = list(map(lambda x: x.strip(), right_path))   # right_path = right_path.strip()

            steering = float(steering)
            throttle = float(throttle)
            brake = float(brake)
            speed = float(speed)

            center_name      = center_path
            left_name        = left_path
            right_name       = right_path

            center_img = cv2.imread(src_path + center_name)
            left_img = list(map(lambda x: cv2.imread(src_path + x), left_name))  # left_img   = cv2.imread(src_path + left_name)
            right_img = list(map(lambda x: cv2.imread(src_path + x), right_name))  # right_img  = cv2.imread(src_path + right_name)

            copyfile(src_path + center_name, dst_path + center_name)
            list(map(lambda x: copyfile(src_path + x, dst_path + x), left_name))   # copyfile(src_path + left_name, dst_path + left_name)
            list(map(lambda x: copyfile(src_path + x, dst_path + x), right_name))  # copyfile(src_path + right_name, dst_path + right_name)

            center_name_flip = center_path.split('.')[0] + '_flip.jpg'
            left_name_flip = list(map(lambda x: x.split('.')[0] + '_flip.jpg', left_name))      #left_name_flip = left_path.split('.')[0] + '_flip.jpg'
            right_name_flip = list(map(lambda x: x.split('.')[0] + '_flip.jpg', right_path))  #right_name_flip = right_path.split('.')[0] + '_flip.jpg'

            center_img_flip = np.fliplr(center_img)
            left_img_flip = list(map(lambda x: np.fliplr(x), left_img))     #left_img_flip   = np.fliplr(left_img)
            right_img_flip = list(map(lambda x: np.fliplr(x), right_img))   # right_img_flip  = np.fliplr(right_img)

            cv2.imwrite(dst_path+center_name_flip, center_img_flip)
            list(map(lambda x,y: cv2.imwrite(dst_path + x, y),left_name_flip, left_img_flip)) #cv2.imwrite(dst_path+left_name_flip, left_img_flip)
            list(map(lambda x, y: cv2.imwrite(dst_path + x, y), right_name_flip, right_img_flip))  #cv2.imwrite(dst_path+right_name_flip, right_img_flip)

            '''
            image_paths = (
                dst_path + center_name,
                dst_path + left_name,
                dst_path + right_name,

                dst_path + left_name_flip,
                dst_path + right_name_flip
            )
            ''';

            image_paths = [
                dst_path + center_name,
                dst_path + center_name_flip
            ]
            left_image_path = list(map(lambda x: dst_path + x, left_name))
            right_image_path = list(map(lambda x: dst_path + x, right_name))
            image_paths.extend(left_image_path)
            image_paths.extend(right_image_path)

            left_image_flip_path = list(map(lambda x: dst_path + x, left_name_flip))
            right_image_flip_path = list(map(lambda x: dst_path + x, right_name_flip))
            image_paths.extend(left_image_flip_path)
            image_paths.extend(right_image_flip_path)

            '''
            stearings = (
                steering,
                steering + correction,
                steering - correction,
                -steering,
                -(steering + correction),
                -(steering - correction)
            )
            ''';

            stearings = [
                steering,
                -steering
            ]
            left_steering_correction = list(map(lambda x: steering + (correction + (x/10)), range(len(left_image_path))))
            right_steering_correction = list(map(lambda x: steering - (correction + (x/10)), range(len(right_image_path))))
            stearings.extend(left_steering_correction)
            stearings.extend(right_steering_correction)

            left_steering_correction_flip = list(map(lambda x: -x, left_steering_correction))
            right_steering_correction_flip = list(map(lambda x: -x, right_steering_correction))
            stearings.extend(left_steering_correction_flip)
            stearings.extend(right_steering_correction_flip)


            for i in range(len(image_paths)):
                train_data.append([
                    image_paths[i],
                    stearings[i],
                ])


            if cnt % 100 == 0:
                print('Parsed ',cnt)
            cnt += 1
            if limit is not None and cnt > limit:
                break


        with open(train_path,'wb') as picklefile:
            pickle.dump(train_data,picklefile)

        with open(train_path+'.json','w') as jsonfile:
            json.dump(train_data,jsonfile)

    return




def loadDataset(name):
    dataset = None
    with open(datasets_path + name + '/' + trainfile, 'rb') as picklefile:
        dataset = pickle.load(picklefile)
    return dataset


def loadBottleneckGenerators(name, model_name, split=0.2 , batch_size=default_batch_size ):
    return loadDatasetGenerators(name+'/'+model_name)

def loadDatasetGenerators(name, split=0.2 , batch_size=default_batch_size):  # return generator
    dataset = loadDataset(name)
    sample = loadData(dataset[0][0]).shape
    dataset_train, dataset_valid = train_test_split(dataset,test_size=split)


    info = {
        'n_train':len(dataset_train),
        'n_train_batch': math.ceil(len(dataset_train)/batch_size),
        'n_valid':len(dataset_valid),
        'n_valid_batch': math.ceil(len(dataset_valid)/batch_size),
        'n_dataset':len(dataset),
        'input_shape':sample
    }
    print(dataset_train)

    return datasetGenerator(dataset_train, batch_size), datasetGenerator(dataset_valid, batch_size), info

def datasetGenerator(dataset, batch_size=default_batch_size):
    n_train = len(dataset)
    while 1:
        shuffle(dataset)
        for offset in range(0, n_train, batch_size):
            dataset_batch = dataset[offset:offset + batch_size]
            images = []
            angles = []
            for sample in dataset_batch:
                X = loadData(sample[0])
                y = sample[1]
                images.append(X)
                angles.append(y)

            images = np.array(images)
            angles = np.array(angles)
            yield shuffle(images,angles)


def loadData(path):
    value=None
    ext = path.split('.')[-1]
    if ext == 'jpg':
        value = cv2.imread(path)
    elif ext == 'p':
        with open(path, 'rb') as pfile:
            value = pickle.load(pfile)
    return value


def recordingList(search="*"):
    os.chdir(recording_path)
    paths = glob.glob(search+os.path.sep)
    os.chdir('../')
    paths=list(filter(lambda x: not x.startswith('_'), paths))
    paths=list(map(lambda x: x[:-1],paths))  #drop folder slash
    return paths

def datasetList(search="*"):
    os.chdir(datasets_path)
    paths = glob.glob(search+os.path.sep)
    os.chdir('../')
    paths=list(filter(lambda x: not x.startswith('_'), paths))
    paths = list(map(lambda x: x[:-1],paths))  # drop folder slash
    return paths