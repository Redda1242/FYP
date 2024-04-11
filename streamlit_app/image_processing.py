from keras.models import load_model
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# Load trained models for different types of image blurring
classifier_model = load_model('model_trained/classifier.keras')
defocus_model = load_model('model_trained/defocus.keras')
motion_model = load_model('model_trained/motion.keras')
camera_shake_model = load_model('model_trained/camera_shake.keras')
box_blur_model = load_model('model_trained/box_blurred.keras')

def enhance_image(image, enhancement_type, enhancement_factor):
    # Dictionary of available image enhancement methods
    enhancers = {
        'contrast': ImageEnhance.Contrast,
        'sharpness': ImageEnhance.Sharpness,
        'brightness': ImageEnhance.Brightness
    }
    # Fetch the appropriate enhancer and apply enhancement
    enhancer = enhancers.get(enhancement_type, ImageEnhance.Contrast)(image)
    return enhancer.enhance(enhancement_factor)

def preprocess_image(image, target_size=(224, 224)):
    # Convert image to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize image to the target size
    image = image.resize(target_size)
    # Convert image data to numpy array and normalize pixel values
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def classify_image(image):
    # Preprocess the image for model input
    preprocessed_image = preprocess_image(image)
    # Predict the class of the image using the classifier model
    predictions = classifier_model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)
    # List of class names corresponding to the prediction outputs
    class_names = ['defocus', 'motion', 'sharp', 'box_blurred', 'camera_shake_blurred']
    return class_names[predicted_class[0]]

def deblur_and_enhance_image(original_image, blur_type, enhancement_type, enhancement_factor):
    # Preprocess the image for model input
    preprocessed_image = preprocess_image(original_image)
    # Mapping of blur types to their respective models
    model_map = {
        'defocus': defocus_model,
        'motion': motion_model,
        'box_blurred': box_blur_model,
        'camera_shake_blurred': camera_shake_model,
    }
    # Apply the appropriate deblurring model if the blur type is recognized
    if blur_type in model_map:
        deblurred = model_map[blur_type].predict(preprocessed_image)
        # Convert numpy array back to image
        processed_image = Image.fromarray((deblurred.squeeze() * 255).astype(np.uint8))
    else:
        # If blur type is not recognized, use the original image
        processed_image = original_image.copy()
    # Apply enhancements to the processed image
    enhanced_image = enhance_image(processed_image, enhancement_type, enhancement_factor)
    # Resize the enhanced image back to original dimensions
    original_size = original_image.size
    return enhanced_image.resize(original_size, Image.Resampling.LANCZOS)

# Function to provide blur tips based on blur type
def blur_tips(blur_type):
    tips = {
        "defocus": "Defocus blur occurs when the camera is out of focus. Adjust the focus settings on your camera, or use autofocus to correct this.",
        "motion": "Motion blur results from camera or subject movement. Increase the shutter speed to capture sharper images.",
        "sharp": "Your image is sharp. No deblurring needed!",
        "box_blurred": "Box blur is typically used in post-processing. Consider reducing this effect in your image editing software.",
        "camera_shake_blurred": "Camera shake causes blurry images. Use a tripod or a higher shutter speed to stabilize your shots."
    }
    return tips.get(blur_type, "No specific tips available for this type of blur.")