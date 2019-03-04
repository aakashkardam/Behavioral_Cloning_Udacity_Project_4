import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D
import matplotlib.pyplot as plt

### Read the data from the csv file
lines = []
#with open('/home/workspace/CarND-Behavioral-Cloning-P3/data_new/driving_log.csv') as csvfile:
with open('/home/workspace/CarND-Behavioral-Cloning-P3/data_new/driving_log.csv') as csvfile:#use this in GPU mode
#with open('/home/workspace/CarND-Behavioral-Cloning-P3/data_1/driving_log.csv') as csvfile:# use this in CPU mode
  next(csvfile)
  render = csv.reader(csvfile)
  for line in render:
    lines.append(line)

lines1 = []
with open('/home/workspace/CarND-Behavioral-Cloning-P3/Additional_Data_turns/driving_log.csv') as csvfile2:
  render = csv.reader(csvfile2)
  for line in render:
    lines1.append(line)
 
  

#print(len(lines))
#print(lines[0])

images = [] #----------------------------- stores the training images (features)                 
measurements = [] #----------------------- stores the steering angle (label)
for line in lines:
  for i in range(3): # using multiple cameras. try this later
    source_path=line[i]
    #print(source_path)
    filename = source_path.split('/')[-1]
    current_path = 'data_new/IMG/'+filename # use this in GPU mode
    #current_path = 'data_1/IMG/'+filename # use this in CPU mode
    image = cv2.imread(current_path)
    #print(image)
    images.append(image)
    measurement = float(line[3])
    if (i==1):
      measurements.append(measurement + 0.05)
    elif (i==2):
      measurements.append(measurement - 0.05)
    else:
      measurements.append(measurement)


for line in lines1:
  for i in range(3): # using multiple cameras. try this later
    source_path=line[i]
    #print(source_path)
    filename = source_path.split('/')[-1]
    current_path = 'Additional_Data_turns/IMG/'+filename # use this in GPU mode
    #current_path = 'data_1/IMG/'+filename # use this in CPU mode
    image = cv2.imread(current_path)
    #print(image)
    images.append(image)
    measurement = float(line[3])
    if (i==1):
      measurements.append(measurement + 0.05)
    elif (i==2):
      measurements.append(measurement - 0.05)
    else:
       measurements.append(measurement)






### Data AUgmentation : flip images
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image,1))
  augmented_measurements.append(measurement*-1.0)

### Training the network
X_train = np.array(augmented_images) #-------------- converts the image data into a numpy array 
y_train = np.array(augmented_measurements) #-------- converts the steering data into a numpy array
#print(X_train)
#print(X_train.shape)
#plt.hist(y_train)
#plt.show()
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
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
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=5)
model.save('model_regression_Udacity_data5.h5')
