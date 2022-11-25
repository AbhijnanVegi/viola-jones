from integral_image import integral_image
from filters import *
import numpy as np

class WeakClassifier():
    def __init__(self, feature, threshold, polarity) -> None:
        self.feature = feature
        self.threshold = threshold
        self.polarity = polarity

    def predict(self, iimg):
        score = self.feature.apply(iimg)
        return 1 if self.polarity * score < self.polarity * self.threshold else 0

class RapidObjectDetector:
    def __init__(self, layers):
        self.layers = layers
        self.X_ff = None
        self.y = None

    def train(self, X, y, rounds=10) :
        h = X.shape[0]
        w = X.shape[1]

        # Create integral image
        X_iimg = [integral_image(x) for x in X]
        X_ii = np.array(X_iimg)

        features = build_features(h, w)
        X__ff = np.array([apply_features(x, features) for x in X_ii])
        self.X_ff = np.array(X__ff)
        self.y = y

        # Initialise AdaBoost weights
        positives = np.count_nonzero(y)
        negatives = len(y) - positives
        
        weights = np.zeros(len(y))

        for i in range(len(y)):
            if y[i] == 1:
                weights[i] = 1 / (2 * positives)
            else:
                weights[i] = 1 / (2 * negatives)

        print(positives, negatives)

    def _train_weak(self, X_ff, y, weights, features):
        
        pos, neg = 0,0
        
        for i, y in enumerate(y):
            if y == 1:
                pos += weights[i]
            else:
                neg += weights[i]

        clfs = []

        for i, feature in enumerate(features):

            curr_feature_scores = []
            for j, x in enumerate(X_ff):
                curr_feature_scores.append((x[i], y[j], weights[j]))

            min_error = float('inf')
            best_threshold = 0
            best_polarity = 0
            best_feature = None

            posw, negw = 0,0
            poss, negs = 0,0

            for x, y, w in sorted(curr_feature_scores, key=lambda x: x[0]):
                error = min(negw + pos - posw, posw + neg - negw)
                if error < min_error:
                    min_error = error
                    best_threshold = x
                    best_polarity = 1 if poss > negs else -1
                    best_feature = feature

                if y == 1:
                    posw += w
                    poss += 1
                else:
                    negw += w
                    negs += 1
            
            clfs.append(WeakClassifier(best_feature, best_threshold, best_polarity))

        return clfs

    def _best_feature(self, clfs, weights):
        best_clf, best_error, best_acc = 0, float('inf'), 0

        for i, clf in enumerate(clfs):
            err, acc = 0, []
            assert(self.X_ff != None)

            for j, x in enumerate(self.X_ff):
                chk = abs(clf.predict(x) - self.y[j])
                acc.append(abs(chk))
                err += weights[j] * chk
            
            avg_error = err/len(weights)

            if avg_error < best_error:
                best_clf, best_error, best_acc = clf, err, acc
                
        return best_clf, best_error, best_acc








    
        