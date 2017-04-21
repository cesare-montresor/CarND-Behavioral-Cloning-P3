import dataset as ds
import model as md

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.utils import plot_model
from keras.models import load_model

model_path ='./models/final_model.h5'
model = load_model(model_path)
plot_model(model, to_file='final_model.png')
exit()


img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
while True:
    f, plts = plt.subplots(4, figsize=(10,10))
    steering = 0
    steering1 = steering

    img1 = ds.randomBrightness(img)
    img2 = ds.randomShadows(img)
    img3, steering1 = ds.randomShift(img, steering)


    plts[0].imshow(img)
    plts[1].imshow(img1)
    plts[2].imshow(img2)
    plts[3].imshow(img3)

    plt.xlabel("Steering: " + str(steering) + " => " + str(steering1) )
    plt.show()

