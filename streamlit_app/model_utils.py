import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from dblur.default.restormer import deblur_single_img

# Model paths - update these paths according to your project structure
# Load models
classifier_model = load_model('model_trained/classifier.keras')
defocus_model = load_model('model_trained/defocus.keras')
motion_model = load_model('model_trained/motion.keras') # Assume typo fixed
camera_shake_model = load_model('model_trained/camera_shake.keras') # Assume typo fixed
box_blur_model = load_model('model_trained/box_blur.keras') # Assume typo fixed


def deblur_and_enhance_image(original_image, blur_type):
    # Preprocess the image for the model without changing the original image
    preprocessed_image = preprocess_image(original_image)
    
    # Apply the model to the preprocessed image
    model_map = {
        'defocus': defocus_model,
        'motion': motion_model,
        'box_blurred': box_blur_model,
        'camera_shake_blurred': camera_shake_model,
    }
    if blur_type in model_map:
        #deblurred = model_map[blur_type].predict(preprocessed_image)
        deblurred = original_image.filter(ImageFilter.UnsharpMask(radius=3, percent=300))
        deblurred = deblurred.filter(ImageFilter.EDGE_ENHANCE_MORE)
        deblurred = deblurred.filter(ImageFilter.DETAIL)
        preprocessed_image = preprocess_image(deblurred)
        deblurred = model_map[blur_type].predict(preprocessed_image)
        # Convert model output back to an image, note that this might not match your exact model output handling
        processed_image = Image.fromarray((deblurred.squeeze() * 255).astype(np.uint8))
    else:
        # If no blur detected or cannot process, use the original image
        processed_image = original_image.copy()

    # Enhance the image based on the user's selection
    #enhanced_image = enhance_image(processed_image, enhancement_type, enhancement_factor)

    # Resize enhanced image to match original dimensions for consistent display
    original_size = original_image.size  # Get original image dimensions
    processed_image_resized = processed_image.resize(original_size, Image.Resampling.LANCZOS)

    return processed_image_resized

def enhance_image(image, enhancement_type='contrast', enhancement_factor=1.1):
    enhancers = {
        'contrast': ImageEnhance.Contrast,
        'sharpness': ImageEnhance.Sharpness,
        'brightness': ImageEnhance.Brightness
    }
    enhancer = enhancers.get(enhancement_type, ImageEnhance.Contrast)(image)
    enhanced_image = enhancer.enhance(enhancement_factor)
    return enhanced_image

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def classify_image(image):
    preprocessed_image = preprocess_image(image)
    predictions = classifier_model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)
    class_names = ['defocus', 'motion', 'sharp', 'box_blurred', 'camera_shake_blurred']
    return class_names[predicted_class[0]]

