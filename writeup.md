# **Behavioral Cloning** 

## Writeup


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeupimage/model.png "Model Visualization"
[image2]: ./writeupimage/T1_Org.jpg "Track1 Orginal"
[image3]: ./writeupimage/T2_Org.jpg "Track2 Orginal"
[image4]: ./writeupimage/R1_1.jpg "Recovery Image"
[image5]: ./writeupimage/R1_2.jpg "Recovery Image"
[image6]: ./writeupimage/R1_3.jpg "Recovery Image"
[image7]: ./writeupimage/Normal.jpg "Normal Image"
[image8]: ./writeupimage/Flip.jpg "Flipped Image"
[image9]: ./writeupimage/crop.jpg "Crop Image"
[image10]: ./writeupimage/Trainset.png "Trainset Image"
[image11]: ./writeupimage/Validationset.png "Validationset Image"
[image12]: ./writeupimage/Truncate.png "Truncate Image"
[image13]: ./writeupimage/Loss.png "Loss Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I have used Nvidia architecture shared in lecture (https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
Changes I made is drop out layer and activation function ELU

#### 2. Attempts to reduce overfitting in the model
* Model has dropout layer.
* Model is fit with data from both the track.
* Model is fit with data augmentation with random flip, randome birghtness contrast, random left,right,center.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).
Note that I tried with different learning rate and optimzer with adam but default value of 0.01 gave me best result.

#### 4. Appropriate training data

* Training data was chosen to keep the vehicle driving on the road.
* I used a combination of center lane driving, recovering from the left and right sides of the road.
* I also capture data for track 2 with sharp turns, straight four lane with baricade, U turns, up and down of roads.
* For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I went through all the model paper.

I selected Nvidia model https://devblogs.nvidia.com/deep-learning-self-driving-cars/ because it was very close to use case. I haven't tried other model due to time constrain of this term, but will like to try resnet other in future.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (0.8 and 0.2 ratio). 

In first attempt, My car was good in driving straight road. But, on few turns it got confuse and went on grass.
* To fix it, I added dropout layer, which worked very well on track 1 (Lake)
Further, I face lot of issues on Track 2. To address it:
* When I tried on Track2 (Mountain), I observe that model was not able to sharp turns, intial four lane road with divider etc was causing car to move toward divider.
* I improved by adding more data from track2 specially for sharp turn. It improved but still could not complete the track2 successful.
* Afterwards, I observe that my data was very skewed to zero steering angle.
* I thought of removing samples with zero angle. Intially, I plan to remove sample at time of load csv file. But later I realize it improves but it fail to work on straight road at some locations like bridge etc.
* To fix issue, I randomly remove different samples with steering angle=0 for each epoch in generator. And, bingo!!. It makes my car drive on track 2 fully without getting stuck or getting fell down from mountain. (Still at one place while turning, it goes little off road and return back on road. but I believe it can also fix with more data, My dataset was very limited with only ten of thousand samples including track1 and track2 data)

#### 2. Final Model Architecture

The final model architecture (model.py :: CreateModel, lines 76-96) consisted of a convolution neural network as shown in visualization.

Here is a visualization of the architecture

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from off the road.

![alt text][image4]

![alt text][image5]

![alt text][image6]

Then I repeated this process on track two in order to get more data points. It also enable me to handle sharp turns and up and down of hilly area.
I also trained in reverse direction.

To augment the data set, 
* I flipped images and angles

![alt text][image7]

![alt text][image8]

* I change brightness and contrast
* I selected left, right and center images.

After the collection process, I have below data
Total Train set 13955
Total Validation set 3489

I preprocess with
* Crop image from top to avoid trees etc. and crop dashboard. I resize crop image to 66x200.

![alt text][image7]

![alt text][image9]

* Normalize image to -1 to 1

I finally randomly shuffled the data for each epoch. 

* As mention earlier in design approach bullet, I also remove few data with zero steering angle as it was biasing my model to steering angle zero.

![alt text][image10]

![alt text][image11]

![alt text][image12]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

For track 1, my model was overfitting and use to train 10-20 epochs.
For track 2, it still seems to underfit( Request reviewer to confirm). Here is the plot of loss function with training with both data set.
![alt text][image13]

Videos:
track1.mp4
<video width="320" height="160" controls>
  <source src="track1.mp4" type="video/mp4">
</video>

track2.mp4
<video width="320" height="160" controls>
  <source src="track2.mp4" type="video/mp4">
</video>

Scope of improvement:
1. More data set from track 2
2. Training with speed and throttle for track2
3. Add L2 or batch normalization.
4. Final model is slow in fitting with epoch.