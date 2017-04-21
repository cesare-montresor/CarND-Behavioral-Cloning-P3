# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[final_model]:      ./writeup_images/final_model.png
[cameras]:      ./writeup_images/cameras.png
[waypoint_jungle_single]:      ./writeup_images/waypoint_jungle_single.png
[CarAI]:   ./writeup_images/CarAI.png
[augment]: ./writeup_images/augment.jpg
[shadows]: ./writeup_images/shadows.jpg
[video]: ./writeup_images/video_preview.png

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py`  
Is the main pipeline, from recordings to a fully trained model
* `dataset.py`  
Contains all the code necessary to deal with recordings and datasets  
* `modellist.py`  
Contains the models used and tested
* `drive.py`  
For driving the car in autonomous mode 
* `writeup_report.md`  
Summarizing the results  
* `model.h5`  
Containing a trained convolution neural network
* `video.mp4`  
[Video of the recorded lap](video.mp4)



* `recordings/`   
Recordings folder
* `datasets/`     
Datasets folder
* `models/`    
Models folder
* `videos/`   
Videos folder


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

The script by default loads './models/final_model.h5'

```sh
python drive.py 
```

Buy any other model can still be used:

```sh
python drive.py ./models/any_other_model.h5
```

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network.
The file, if needed, parse a recordings folder and consolidate it into a new dataset having the same name.
Then, it load the desired architecture (see model.py) and eventually preload weight from previous runs.
Out of the dataset 2 separate generators are produced, along with additional information regarding the dataset.
The model is then trained.
Two callbacks have been added, one for saving a chackpoint after every epoch (in case of improvments) and
a second one to stop the training in case in the last N epochs there have been no noticable imporvments in the val_loss

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

After testing several models like LeNet, Inception (FE, retrin), the model from P2 and the nvidia one.  
I realized that for the given task and with a sufficient amount of data I would a probably managed to use any and have a good result.  
I ended up choosing a modified version of the nvidia network with Relu as non-linear activation and I added dropout for regularization purposes and prevent overfitting.   

#### 2. Attempts to reduce overfitting in the model

The model was trained using separate training a validation sets, using a 0.2 split.  
The val_loss was monitored and a checkpoints is saved after every epoch in case of improvements.  
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.  
As mention above I added 2 dropout layer in between the FC to help prevent overfitting.
The dataset have been augmented (see below)

#### 3. Model parameter tuning

In the beginning I choose adam as optimizer by default, but later I had so issue training my model (...I thought, later I found out I was looking at the wrong metrics)   
so, among other attempts, I've tried to change the optimization algorithms I noticed that sgd was converging much faster.   
I wish I knew the numerical reasons behind it but unfortunatly I didn't had the proper time to have an in-depth look and compare the various optimizers.  

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road. 
The combined data have been augmented by flipping the images to balance the (left) steering.
To the resulting data further augmentation have been applied to help it generalize better for:
* vertical and horizontal shifts (311:dataset.py)
* change in brightness (320:dataset.py)
* shadows (low aplha) and occlusions (high alpha) (330:dataset.py)

![augment]

As it was suggested to collect additional driving data using the simulator, but knowing that I'm quite poor driver 
when it comes to videogames, I instead decided to leverage my IT skill and get the simulator to drive using waypoints, 
while recording thinking that, after all, is a technique that can be implemented also outside a simulator.  
Especially because nor C# or Unity are my strong skill, it came as a nice surprise a pre-made waypoint rail for the lake track and the CarAI script almost ready to go.

![CarAI]

So I modified the necessary scripts and parameters and adjusted the rails until I was satisfied. As I was feeling motivated I added a rail also for the jungle track (single lane).

![waypoint_jungle_single]

As I was feeling like experimenting I've been adding also 3 extra cameras per side of the car and added those in the recording too.
Even if at first having extra cameras sounded like a good idea ultimetly I didn't use it to train my final model, perhaps having a more refined steering correction mechanisms might deliver better results.
However, you will still find traces in the code when consolidating the dataset, is possible to load multiple side cameras.

![cameras]

 _Possible improvments on the simulator:_ 
* Adding a waypoint recording tool in the simulator, to be able to record and playback your own rails.
* Integrating the recorder with the CarAI script:  
  * Generate recovery laps by pausing the recorder before it wanders and restart when it corrects, decreasing the need of hand/hardcoded corrections.
  * Marking each frame with it's waypoint it could later simplify the job of uniforming the dataset in terms of scenarios, turns, etc.

In case you might be interested into the code of the simulator is available on github:

https://github.com/cesare-montresor/self-driving-car-sim

_I hope you don't mind I tampered with the simulator._

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started by trying out on feature extraction and retrain on InceptionV3 and it was way to long. So I've tried the model
I've used in the P2 but with bigger images it turn out to be much bigger and harder to train but not much more effective than
than LeNet or Nvidia, so after few attempts I've seattle for a slightly modified version of the nvidia NN with dropout.  

#### 2. Final Model Architecture

The final model architecture (23:model.py) is pretty much like it appears in the slides but with added 0.5 of dropout in between the FC layers, here is an extract:

```python
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
        print('Loading weights: YES', load_weight)
        model.load_weights(load_weight)
    else:
        print('Loading weights: NO', load_weight)

    return model
```

A simple visualization can be obtained by using the deafult keras tools:

```
model_path ='./models/final_model.h5'
model = load_model(model_path)
plot_model(model, to_file='final_model.png')
```


![final_model]

#### 3. Creation of the Training Set & Training Process

I create a few datasets using the technique of waypoint-driving described above, then using is possible `loadDatasetGenerators(['ds1','ds2'])` to load generators producind data from multiple datasets at once.  
In the beginning I started by using multiple side cameras and later ditched the idea and I used a simple left,center,right. 
Focusing instead on augmenting the dataset with the horizontal shift to compensate my poor steering correction tecnique and how to improve it.  
I thought of adding occlusions and shadows particularly while I was placing the waypoint for the jungle track from birdeye view, I could really appreciate how many different lighting conditions, shadows and sharp changes in brightness there are.  
![shadows]

#### Personal considerations

I wish I had more time to try out other variants especially, the seering correction for left and right images (with multiple side cameras).
Train the model to predict also the throttle and break, to be able to drive slow when is required and fast when is possible, giving a much more natural feel and more precision in turns.
One of the ideas I would like to try the most would be to augment the Y to predict not just the current steering but also N future frames, the rational behind it is that the current frame does already contain the "future frames" within and a human driver constantly taks advantage of that,
the hope would be to help the model to make sense of the image using more relevant part of information as now has to "work harder".
With this project it also changed my prospective on what to feed to a NN, if before I was trying to preprocess and clenup (equalization, etc) the image, 
now I'm starting to realize that is better instead train your model on conditions (much) worst than what you will find and it will end up generalizing much better.
