import csv, cv2, glob, os, math
import numpy as np
import pickle
import numpy as np
from shutil import copyfile
from sklearn.model_selection import train_test_split
from keras.applications import InceptionV3
import h5py, pickle
import datetime
import json
from sklearn.utils import shuffle
import cv2
import os


cvsname = 'driving_log.csv'
trainfile = 'train.p'
recording_path = './recordings/'
datasets_path = './datasets/'
default_batch_size = 32

def standardModelName(name='model'):
    name += +'_'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
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

def recordingToDataset_allCenter(name,correction_left=0.2,correction_right=0.2,force=False,limit=None, reindex_only=False, side_cameras=1):  #performs center
    #print('name',name)
    src_path = recording_path + name + '/'
    dst_path = datasets_path + name + '/'
    train_path = dst_path+trainfile

    os.makedirs(dst_path+'IMG/', exist_ok=True)

    train_data = []

    if force: reindex_only = False

    if os.path.isfile(train_path) and not (force or reindex_only):
        return  #avoid to repeat the import

    cnt = 0
    with open(src_path+cvsname) as csvfile:
        csvreader = csv.reader(csvfile)
        for line in csvreader:
            center_path, left_path, right_path, steering, throttle, brake, speed = line

            #get paths
            center_path = center_path.strip()
            left_path = left_path.split("|")
            left_path = list(map(lambda x: x.strip(), left_path))     # left_path = left_path.strip()
            right_path = right_path.split("|")
            right_path = list(map(lambda x: x.strip(), right_path))   # right_path = right_path.strip()

            #pick number of side cameras
            if side_cameras > len(left_path):
                side_cameras = len(left_path)
            left_path = left_path[0:side_cameras]
            right_path = right_path[0:side_cameras]

            #standardize data type
            steering = float(steering)
            throttle = float(throttle)
            brake = float(brake)
            speed = float(speed)

            #prepare filenames for (center, left, right, center_flip, left_flip, right_flip)
            center_name      = center_path
            left_name        = left_path
            right_name       = right_path
            center_name_flip = center_path.split('.')[0] + '_flip.jpg'
            left_name_flip = list(map(lambda x: x.split('.')[0] + '_flip.jpg',left_name))
            right_name_flip = list(map(lambda x: x.split('.')[0] + '_flip.jpg',right_path))

            #read,copy,flip,write images
            if not reindex_only:
                center_img = cv2.imread(src_path + center_name)
                left_img = list(
                    map(lambda x: cv2.imread(src_path + x), left_name))
                right_img = list(map(lambda x: cv2.imread(src_path + x), right_name))

                copyfile(src_path + center_name, dst_path + center_name)
                list(map(lambda x: copyfile(src_path + x, dst_path + x), left_name))
                list(map(lambda x: copyfile(src_path + x, dst_path + x), right_name))

                center_img_flip = np.fliplr(center_img)
                left_img_flip = list(map(lambda x: np.fliplr(x), left_img))
                right_img_flip = list(map(lambda x: np.fliplr(x), right_img))

                cv2.imwrite(dst_path+center_name_flip, center_img_flip)
                list(map(lambda x,y: cv2.imwrite(dst_path + x, y),left_name_flip, left_img_flip)) #cv2.imwrite(dst_path+left_name_flip, left_img_flip)
                list(map(lambda x, y: cv2.imwrite(dst_path + x, y), right_name_flip, right_img_flip))  #cv2.imwrite(dst_path+right_name_flip, right_img_flip)

            # building image paths array
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


            # build steering array
            stearings = [
                steering,
                -steering
            ]

            left_steering_correction = list(map(lambda x: steering + correction_left, range(len(left_image_path))))
            right_steering_correction = list(map(lambda x: steering - correction_left, range(len(right_image_path))))
            stearings.extend(left_steering_correction)
            stearings.extend(right_steering_correction)

            left_steering_correction_flip = list(map(lambda x: -x, left_steering_correction))
            right_steering_correction_flip = list(map(lambda x: -x, right_steering_correction))
            stearings.extend(left_steering_correction_flip)
            stearings.extend(right_steering_correction_flip)

            # format, consolidate data
            allcenter_data = []
            for i in range(len(image_paths)):
                allcenter_data.append([
                    image_paths[i],
                    stearings[i],
                ])
            train_data.extend(allcenter_data)

            # apply additional augumentation:
            ncopy = 3 # num of copy per each image
            for entry in allcenter_data:
                path = entry[0]
                steering = entry[1]
                for i in range(ncopy):
                    num = str(i)
                    steering_shift = steering
                    # build paths
                    path_brt = filename_append(path, '_brt' + num)
                    path_shift = filename_append(path, '_shift' + num)
                    path_shw = filename_append(path, '_shw' + num)

                    # augument and write images
                    if not reindex_only:
                        img = cv2.imread(path,cv2.COLOR_BGR2RGB)

                        img_brt = randomBrightness(img)
                        cv2.imwrite(path_brt, img_brt)

                        img_shift, steering_shift = randomShift(img, steering)
                        cv2.imwrite(path_shift, img_shift)

                        img_shw = randomShadows(img)
                        cv2.imwrite(path_shw, img_shw)

                    train_data.extend([
                        [path_brt, steering],
                        [path_shift, steering_shift],
                        [path_shw, steering]
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

def loadDataset(names):
    dataset = []
    if type(names) == type(""):
        names = [names]

    for name in names:
        with open(datasets_path + name + '/' + trainfile, 'rb') as picklefile:
            dataset.extend(pickle.load(picklefile))

    return dataset

def loadDatasetGenerators(name, split=0.2 , batch_size=default_batch_size, limit=None, processImages=None, equalize=False):  # return generator
    dataset = loadDataset(name)
    if limit is not None:
        dataset = dataset[0:limit]
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

    return datasetGenerator(dataset_train, batch_size,processImages=processImages, equalize=equalize), datasetGenerator(dataset_valid, batch_size, processImages=processImages, equalize=equalize), info

def datasetGenerator(dataset, batch_size=default_batch_size, processImages=None, equalize=False):
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

            if processImages is not None:
                images = processImages(images)

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


def randomBrightness(img, limit=0.4):
    img_new = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img_new = np.array(img_new, dtype = np.float64)
    img_new[:,:,2] = img_new[:,:,2] * (np.random.uniform(low=limit, high=2-limit))
    img_new[:,:,2][img_new[:,:,2]>255] = 255 #cap values
    img_new = np.array(img_new, dtype = np.uint8)
    img_new = cv2.cvtColor(img_new,cv2.COLOR_HSV2RGB)
    return img_new

def randomShift(img, steering, max_shift_x = 10, max_shift_y = 10, steering_strenght=1):
    height, width, depth = img.shape
    x_seed = np.random.uniform(-1,1)
    deltaX = max_shift_x * x_seed
    deltaY = max_shift_y * np.random.uniform(-1,1)
    steering += x_seed * steering_strenght
    trans = np.float32([[1, 0, deltaX], [0, 1, deltaY]])
    img_new = cv2.warpAffine(img, trans, (width,height))
    return img_new, steering

def randomShadows(img, max_shadows = 3, min_aplha=0.1, max_aplha=0.8, min_size=0.2, max_size=0.8 ):
    img_new = img.copy()
    height, width, depth = img_new.shape
    # print(width,height)
    shadow_num = int(max_shadows * np.random.uniform())+1
    for i in range(shadow_num):
        x = int(width * np.random.uniform())
        y = int(height * np.random.uniform())
        w2 = int( (width * np.random.uniform(min_size,max_size))/2 )
        h2 = int( (height * np.random.uniform(min_size,max_size))/2 )
        top, bottom = y - h2, y + h2
        left, right = x - w2, x + w2
        top, bottom = max(0, top), min(height, bottom)
        left, right = max(0, left), min(width, right)
        img_new[top:bottom, left:right, :] = img_new[top:bottom, left:right, :] * np.random.uniform(min_aplha,max_aplha)
    return img_new


def histogramEqualizationAndColorSpace(image):
    ycrcb=cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    return ycrcb

def processImage(image):
    image = histogramEqualizationAndColorSpace(image)
    return image

def processImages(images):
    images = [processImage(image) for image in images]
    return images



def filename_append(path, suffix):
    parts = path.split(".")
    ext = parts[-1]
    base = ".".join(parts[:-1])+suffix+'.'+ext
    return base