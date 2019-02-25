import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout
import cv2
import sklearn
import matplotlib.image as mpimg
from random import randint
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import losses
from keras.utils import plot_model

# Model class to create model with training set.
class Model:
    
    def __init__(self, epochs=100, batch_size=48, keep_prob=0.8):
        self.epochs = epochs
        self.trainSample = None
        self.validSample = None
        self.model = None
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.data_distribution = np.zeros((26,), dtype=int)
        self.samples = []

    # function to plot histogram of sample distribution
    def plotHist(self, sample, name):
        plt.clf()
        hist = []
        for s in sample:
            hist.append(np.float(s[3]))
        plt.hist(hist,25)
        plt.title(name)
        plt.ylabel('Number Data sample')
        plt.xlabel('Sampled steering angles')
        plt.savefig(name + '.png')


    # Get the histogram  of samples distribution.    
    def getHist(self, sample):
        hist = []
        for s in sample:
            hist.append(np.float(s[3]))
        [hist, val] = np.histogram(np.array(hist), bins=25)
        return hist

    # Load the dataset and append to samples list
    def loadDataset(self, path):
        lines = []
        with open((path + '/driving_log.csv')) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)        #skip headers lines
            for line in reader:
                line[0] = path + '/IMG/'+line[0].split('/')[-1]
                line[1] = path + '/IMG/'+line[1].split('/')[-1]
                line[2] = path + '/IMG/'+line[2].split('/')[-1]
                lines.append(line)
        print('Data length', len(lines))
        self.samples.extend(lines)

    # Split the dataset in train and test
    def split(self):

        self.trainSample, self.validSample = train_test_split(self.samples, test_size=0.2)
        self.plotHist(self.trainSample, 'Train set')
        self.plotHist(self.validSample, 'Validation set')
        print('Total Train set', len(self.trainSample))
        print('Total Validation set', len(self.validSample))
        reduce_sample = self.reduce_zero_sample(self.trainSample)
        self.plotHist(reduce_sample, 'Approximate distribution for each epoch')

    # Craete model with keras. (https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
    def createModel(self):
        model = Sequential()
        # Normalize between -1 to 1, Shape is cropped to size 66x200 YUV image
        model.add(Lambda(lambda x:  (x / 128) - 1., input_shape=(66, 200, 3)))
        model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='elu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='elu'))

        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dropout(self.keep_prob))
        model.add(Dense(50, activation='elu'))
        model.add(Dropout(self.keep_prob))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))  
        model.summary()
        # Default adam optimizer.
        model.compile(loss=losses.mean_squared_error, optimizer='adam')
        self.model = model
        plot_model(model, to_file='model.png')
        return model

    # Preprocessing image, Similar function is copied to drive.py for preprocessing.
    def preprocess(self, image):
        image = image[66:-25, :, :]    # Crop non-relevant information of images like trees and car dashboard.
        image = cv2.resize(image, (200, 66), cv2.INTER_AREA) # Resize to nvidia specific image size.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)   # COnvert to YUV as per article.
        return image

    # create augment data
    # 1. random choice between left, right and center. (Note, Selecting left and right for extreme angle causing problem with sharp turn on Track2) 
    # 2. random Flip image and steering angle
    # 3. Random contrast 
    # 4. Random brightness
    def dataAugment(self, batch_sample):
    
        ## Choose between center, left and right
        steering_angle = float(batch_sample[3])
        name = batch_sample[0]
        if ((steering_angle < 0.7) & (steering_angle > -0.7)): 
            choice = randint(0, 2)
            name = batch_sample[choice]
            if choice == 0:
                steering_angle = float(batch_sample[3])
            elif choice == 1:
                steering_angle = float(batch_sample[3]) + 0.2
            else:
                steering_angle = float(batch_sample[3]) - 0.2
        image = mpimg.imread(name)
        choice = randint(0, 1)
        if choice == 1:
            image = image = cv2.flip(image, 1)
            steering_angle = -steering_angle;
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        random_bright = np.random.uniform(low=0.7, high=1.3)
        h = hsv[:,:,2]*random_bright
        random_bright = np.random.uniform(low=-20, high=20)
        h = h+random_bright
        h[h>255]  = 255
        h[h<0]  = 0
        hsv[:,:,2] = h
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return image, steering_angle
      
    # Lot of data with zero sample causes track2 to fail at sharp angle on track2. Reduce samples with zero.
    def reduce_zero_sample(self, samples):
        reduce_samples = samples
        hist = self.getHist(reduce_samples)
        hist = np.sort(hist)
        max = hist[len(hist) - 1]
        max2 = hist[len(hist) - 2]
        reduce = 0.8 * (max - max2)
        count = 0
        # reduce steering = 0 for each iteration to correct distribution
        for line in reduce_samples:
            if ((np.float(line[3]) == 0)):
                count+=1
                reduce_samples.remove(line)
                if (count > reduce):
                    break
        return reduce_samples

    # generator for train and valid samples.
    def generator(self, samples):
        while 1:
            shuffle(samples)
            # Due to skew data set for zero steering angle.
            # We can remove sample with zero steering angle at load time
            # However, it remove few important images permanently from train set.
            # To address it, for each train set pick random data with zero steering angle.
            # This was done specially for track 2. Track 1 can be train without below function call as well.
            reduce_samples = self.reduce_zero_sample(samples)
            num_samples = len(reduce_samples)
            for offset in range(0, num_samples, self.batch_size):
                batch_samples = reduce_samples[offset:offset+self.batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    image, steering_angle = self.dataAugment(batch_sample)       # Want to read RGB image to match drive.py format
                    image = self.preprocess(image) # preprocess image
                    images.append(image)
                    angles.append(steering_angle)
                
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)
    
    # Execute training of model, Select best model with best validation accuracy.
    def run(self):
        train_generator = self.generator(self.trainSample)
        validation_generator = self.generator(self.validSample)
        checkpointer = ModelCheckpoint(filepath='./model1.h5', verbose=1, save_best_only=True)
        history_object = self.model.fit_generator(generator=train_generator,
                                 validation_data=validation_generator,
                                 epochs=self.epochs,
                                 steps_per_epoch=len(self.trainSample)/self.batch_size,
                                 validation_steps=len(self.validSample)/self.batch_size,
                                 verbose=1,
                                 callbacks=[checkpointer])
        plt.clf()
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.savefig('Loss.png')

# Main function for training.
def train(): 
    model = Model()
    model.loadDataset('../Data3/')             # Data set from track 2
    model.loadDataset('../data_Sample/')       # Sample data set for track1
    model.loadDataset('../Data2/')             # Few more sample for track2
    model.split()
    model.createModel()
    model.run()

train()