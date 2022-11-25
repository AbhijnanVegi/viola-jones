from load_image import load_data
from train import RapidObjectDetector

import numpy as np
import cv2

X,y = load_data()

# choose random ints 
ind = np.random.choice(len(X), 100, replace=False)

clf = RapidObjectDetector(5)
clf.train(X[ind],y[ind])