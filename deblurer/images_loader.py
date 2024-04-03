from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Assume you have your dataset in two directories: 'blurred/' and 'sharp/'
# Each directory contains the respective images with the same filenames.
def load_images(directory, target_size=(224, 224)):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        image = load_img(img_path, target_size=target_size)
        image = img_to_array(image) / 255.0  # Normalize to [0, 1]
        images.append(image)
    return np.array(images)
