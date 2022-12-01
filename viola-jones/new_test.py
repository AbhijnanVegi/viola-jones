from load_image import load_train_data, load_test_data
from detector import RapidDetector
import pickle

import numpy as np

X_train, y_train = load_train_data(-1)



clf = RapidDetector([10,10,10,10,10])
clf.train(X_train,y_train)
pickle.dump(clf, open('model.pkl', 'wb'))

# Test training accuracy
tp, tn, fp, fn = 0,  0, 0 , 0

for i in range(len(X_train)):
    if (y_train[i] == 1):
        if (clf.predict(X_train[i]) == 1):
            tp += 1
        else:
            fn += 1
    else:
        if (clf.predict(X_train[i]) == 1):
            fp += 1
        else:
            tn += 1

print("Training Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
print("Training Precision: ", tp / (tp + fp))
print("Training Recall: ", tp / (tp + fn))
print("Confusion matrix: tp : {tp}, tn : {tn}, fp : {fp}, fn : {fn}".format(tp=tp, tn=tn, fp=fp, fn=fn))




    