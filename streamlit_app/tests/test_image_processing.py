import numpy as np
from PIL import Image
from image_processing import enhance_image, preprocess_image, classify_image, blur_tips

def test_blur_tips():
    assert blur_tips("defocus") == "Defocus blur occurs when the camera is out of focus. Adjust the focus settings on your camera, or use autofocus to correct this."
    assert blur_tips("motion") == "Motion blur results from camera or subject movement. Increase the shutter speed to capture sharper images."
    assert blur_tips("sharp") == "Your image is sharp. No deblurring needed!"
    assert blur_tips("nonexistent") == "No specific tips available for this type of blur."
    
def test_enhance_image():
    # Create a simple image with a single color
    img = Image.new('RGB', (100, 100), color = 'red')
    # Test enhancement
    enhanced_img = enhance_image(img, 'brightness', 1.5)
    # Check that the image is still an image
    assert isinstance(enhanced_img, Image.Image)

def test_preprocess_image():
    img = Image.new('RGB', (100, 100), color = 'blue')
    processed_img = preprocess_image(img)
    # Check shape of the processed image array
    assert processed_img.shape == (1, 224, 224, 3)  # Assuming the target size is (224, 224)

def test_classify_image():
    img = Image.new('RGB', (100, 100), color = 'green')
    # Mocking the output of classifier model might require patching if using actual model predictions
    class_name = classify_image(img)
    # Test type of class_name returned
    assert isinstance(class_name, str)
    
