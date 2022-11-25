import cv2
import os
import numpy as np

def load_data(num=50):
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


if __name__=="__main__":
    images, labels = load_data()
    print(images.shape)