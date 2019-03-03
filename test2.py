# import cv2
# from skimage.feature import hog
# im = cv2.imread('/Users/Pulkit_Agarwal-BNG/Assignments/hack2019/Model/data/F/189.jpg')
# fd, hog_image = hog(im, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualise=True,multichannel=True)

# cv2.imshow('hog',hog_image)
# cv2.waitKey(0)

import pickle
import cv2
from skimage.feature import hog
import numpy as np

model = pickle.load(open('model', 'rb'))

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     crop = cv2.line(frame,(511,161),(911,161),(255,0,0),5)
#     crop = cv2.line(crop,(511,161),(511,561),(255,0,0),5)
#     crop = cv2.line(crop,(511,561),(911,561),(255,0,0),5)
#     crop = cv2.line(crop,(911,161),(911,561),(255,0,0),5)
    crop_img = frame[161:561, 511:911]
    # Display the resulting frame
    cv2.imshow('frame',crop_img)
    a = ['A','F','other']

#     hist = hog.compute(crop_img,winStride,padding,locations)
    im = cv2.resize(crop_img, (480,640), interpolation = cv2.INTER_AREA)
    hist, hog_image = hog(im, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True,multichannel=True)
#     hist = np.transpose(hist)
    hist = np.transpose(hist.reshape(9600,1))
#     hist = pca.transform(hist)
    print(model.predict(hist))
    print('class',a[np.argmax(model.predict(hist),axis=1)[0]])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()






