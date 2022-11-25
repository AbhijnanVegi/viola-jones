from load_image import load_train_data, load_test_data
from train import RapidObjectDetector
import pickle
import numpy as np

X,y = load_train_data(100)

clf = RapidObjectDetector(5)
clf.train(X,y)

pickle.dump(clf, open('model.pkl', 'wb'))

X,y = load_test_data(10)

ps, ng  = 0, 0
for i in range(len(X)):

    if (y[i] == clf.predict(X[i])):
        ps += 1
    else:
        ng += 1
    print("Predicted: ", clf.predict(X[i]), "Actual: ", y[i])

print(ps, ng)



