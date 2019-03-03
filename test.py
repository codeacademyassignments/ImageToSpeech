from keras.models import Sequential
from keras.layers import Dense
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import cv2
import numpy as np


model = Sequential()

winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)

model  =  pickle.load(open('model', 'rb'))

pca = pickle.load(open('pca','rb'))

cap = cv2.VideoCapture(0)
a = ['A','F','other']

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    crop = cv2.line(frame,(511,161),(911,161),(255,0,0),5)
    crop = cv2.line(crop,(511,161),(511,561),(255,0,0),5)
    crop = cv2.line(crop,(511,561),(911,561),(255,0,0),5)
    crop = cv2.line(crop,(911,161),(911,561),(255,0,0),5)
    crop_img = crop[161:561, 511:911]
    # Display the resulting frame
    cv2.imshow('frame',crop_img)
    hist = hog.compute(crop_img,winStride,padding,locations)
    hist = np.transpose(hist)
    hist = pca.transform(hist)
    print(model.predict(hist))
    print('class',a[np.argmax(model.predict(hist),axis=1)[0]])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()