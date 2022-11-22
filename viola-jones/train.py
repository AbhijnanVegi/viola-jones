from integral_image import integral_image
from filters import *
import numpy as np

class RapidObjectDetector:
    def __init__(self, layers):
        self.layers = layers

    def train(self, X, y) :
        h = X.shape[0]
        w = X.shape[1]

        # Create integral image
        X_iimg = [integral_image(x) for x in X]
        X_ii = np.array(X_iimg)

        features = build_features(h, w)
        X__ff = np.array([apply_features(x, features) for x in X_ii])
        X_ff = np.array(X_ff)
    
        