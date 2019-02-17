import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

### Read the data from the csv file
lines = []
with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
  render = csv.reader(csvfile)
  for line in render:
    lines.append(line)
    #print(line[0],"" ,line[3]) 
  

print(len(lines))

images = [] #----------------------------- stores the training images (features)                 
measurements = [] #----------------------- stores the steering angle (label)
for line in lines:
  image = cv2.imread(line[0])
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

### Training the network
X_train = np.array(images) #-------------- converts the image data into a numpy array 
y_train = np.array(measurements) #-------- converts the steering data into a numpy array

model = Sequential()
model.add(Flatten(input_shape=(len(lines),640,480,3)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True)
model.save('model_regression.h5')
