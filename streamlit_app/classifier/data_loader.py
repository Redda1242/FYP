import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

def load_images(directory, size=(224, 224), num_classes=5):
    images = []
    labels = []
    for label, subdir in enumerate(sorted(os.listdir(directory))):
        subpath = os.path.join(directory, subdir)
        for file in os.listdir(subpath):
            img_path = os.path.join(subpath, file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, size)
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
            labels.append(label)
    images = np.array(images)
    labels = to_categorical(labels, num_classes=num_classes)  # Convert labels to one-hot encoding
    return images, labels
