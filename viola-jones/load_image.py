import cv2
import os
import numpy as np

def load_train_data(num=50):
    # open directory
    faces = os.listdir('train/face')

    images = []
    labels = []

    for file in faces[:num]:
        image = cv2.imread('train/face/' + file, -1)
        images.append(image)
        labels.append(1)

    non_faces = os.listdir('train/non-face')

    for file in non_faces[:num]:
        image = cv2.imread('train/non-face/' + file, -1)
        images.append(image)
        labels.append(0)

    return np.array(images), np.array(labels)

def load_test_data(num=50):

    faces = os.listdir('test/face')

    images = []
    labels = []

    for file in faces[:num]:
        image = cv2.imread('test/face/' + file, -1)
        images.append(image)
        labels.append(1)

    non_faces = os.listdir('test/non-face')

    for file in non_faces[:num]:
        image = cv2.imread('test/non-face/' + file, -1)
        images.append(image)
        labels.append(0)

    return np.array(images), np.array(labels)


if __name__=="__main__":
    images, labels = load_train_data()
    print(images.shape)