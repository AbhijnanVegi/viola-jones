from load_image import load_data
from train import RapidObjectDetector
import pickle
import numpy as np

X,y = load_data(3)

clf = RapidObjectDetector(1)
clf.train(X,y)

for i in range(6):
    print(clf.predict(X[i]))