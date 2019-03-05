import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
import matplotlib.image as mpimg

###f,ax=plt.subplots(1,3,figsize=(15,10))
####plt.suptitle('Three random images from the left end camera on the car')
###plt.subplot(131)
###plt.imshow(mpimg.imread('left_2019_02_17_04_19_49_888.jpg'))
###plt.subplot(132) 
###plt.imshow(mpimg.imread('left_2019_02_17_04_19_40_782.jpg'))
###plt.subplot(133) 
###plt.imshow(mpimg.imread('left_2019_02_17_04_19_15_684.jpg'))

###f,ax=plt.subplots(1,3,figsize=(15,10))
####plt.suptitle('Three random images from the left end camera on the car')
###plt.subplot(131)
###plt.imshow(mpimg.imread('center_2019_02_17_04_20_36_990.jpg'))
###plt.subplot(132) 
###plt.imshow(mpimg.imread('center_2019_02_17_04_19_50_239.jpg'))
###plt.subplot(133) 
###plt.imshow(mpimg.imread('center_2019_02_17_04_22_45_329.jpg'))

###f,ax=plt.subplots(1,3,figsize=(15,10))
####plt.suptitle('Three random images from the left end camera on the car')
###plt.subplot(131)
###plt.imshow(mpimg.imread('right_2019_02_17_04_21_29_564.jpg'))
###plt.subplot(132) 
###plt.imshow(mpimg.imread('right_2019_02_17_04_19_50_313.jpg'))
###plt.subplot(133) 
###plt.imshow(mpimg.imread('right_2019_02_17_04_22_45_493.jpg'))

f,ax=plt.subplots(1,3,figsize=(15,10))
#plt.suptitle('Three random images from the left end camera on the car')
plt.subplot(131)
plt.imshow(mpimg.imread('autonomous1.png'))
plt.subplot(132) 
plt.imshow(mpimg.imread('autonomous2.png'))
plt.subplot(133) 
plt.imshow(mpimg.imread('autonomous3.png'))


#plt.subplot(334) 
#plt.subplot(335) 
#plt.subplot(336) 
#plt.subplot(337) 
#plt.subplot(338) 
#plt.subplot(339)
f.savefig('AutonomousModeImages.png',bbox_inches='tight')
plt.show() 
#ax = ax.reshape(-1) # or use ax = ax.ravel()
# Randomly plot some images with their Class IDs
#for a in ax:    
#    i = random.randint(0,n_train)
#    a.imshow(cv2.imread())
#    a.axis("off")
#    a.set_title(y_train[i])
