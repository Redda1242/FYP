from keras.models import load_model
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array

def predict_image_class(img_path):
    # Load trained model
    model_path = 'model_trained/classifier.keras'
    model = load_model(model_path)
    
    # Import image and resize
    image = Image.open(img_path)
    image_resized = image.resize((224, 224))
    image_array = img_to_array(image_resized)
    
    # Normalize the image
    image_array /= 255.0
    
    image_batch = np.expand_dims(image_array, axis=0)
    
    # Make predictions
    predictions = model.predict(image_batch)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)
    
    # Predict based on following classes
    class_names = ['defocus', 'motion', 'sharp', 'box_blurred', 'camera_shake_blurred']  
    predicted_class_name = class_names[predicted_class[0]]
    
    return predicted_class_name
