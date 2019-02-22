import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D

### Read the data from the csv file
lines = []
with open('/home/workspace/CarND-Behavioral-Cloning-P3/data_new/driving_log.csv') as csvfile:
#with open('data_1/driving_log.csv') as csvfile:
  render = csv.reader(csvfile)
  for line in render:
    lines.append(line)
    #print(line[0],"" ,line[3]) 
  

#print(len(lines))
#print(lines[0])

images = [] #----------------------------- stores the training images (features)                 
measurements = [] #----------------------- stores the steering angle (label)
for line in lines:
  source_path=line[0]
  #print(source_path)
  filename = source_path.split('/')[-1]
  current_path = '../data/IMG/'+filename
  image = cv2.imread(current_path)
  #print(image)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

### Training the network
X_train = np.array(images) #-------------- converts the image data into a numpy array 
y_train = np.array(measurements) #-------- converts the steering data into a numpy array
#print(X_train)
#print(X_train.shape)
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=10)
model.save('model_regression_Udacity_data.h5')
