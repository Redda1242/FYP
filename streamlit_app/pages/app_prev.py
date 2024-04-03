import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from dblur.default.restormer import deblur_single_img

# Load models
classifier_model = load_model('model_trained/classifier.keras')
defocus_model = load_model('model_trained/defocus.keras')
motion_model = load_model('model_trained/motion.keras') # Assume typo fixed
camera_shake_model = load_model('model_trained/camera_shake.keras') # Assume typo fixed
box_blur_model = load_model('model_trained/box_blur.keras') # Assume typo fixed

st.set_page_config(page_title="PureView - Image Processing App", page_icon=":camera:", layout="wide")
max_width = 400
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

def deblur_and_enhance_image(original_image, blur_type, enhancement_type, enhancement_factor):
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
    enhanced_image = enhance_image(processed_image, enhancement_type, enhancement_factor)

    # Resize enhanced image to match original dimensions for consistent display
    original_size = original_image.size  # Get original image dimensions
    enhanced_image_resized = enhanced_image.resize(original_size, Image.Resampling.LANCZOS)

    return enhanced_image_resized

def display_blur_info(blur_type):
    blur_info = {
        'defocus': ("Defocus blur occurs when the camera is out of focus. To prevent this, ensure your camera is correctly focused on the subject. For more, visit [PhotographyLife](https://photographylife.com/what-is-defocus).", "Defocus Blur"),
        'motion': ("Motion blur results from the movement of the subject or camera during exposure. Use faster shutter speeds to avoid this. Learn more at [CaptureLandscapes](https://www.capturelandscapes.com/how-to-avoid-motion-blur/).", "Motion Blur"),
        'camera_shake_blurred': ("Camera shake blur happens due to the camera moving while the shutter is open. Stabilize the camera with a tripod or use a faster shutter speed. More at [Digital Photography School](https://digital-photography-school.com/how-to-avoid-camera-shake/).", "Camera Shake Blur"),
        'box_blurred': ("Box blur is a post-processing effect and not typically caused by camera settings. It's used creatively or to blur parts of an image in editing software. Learn about its applications at [Wikipedia](https://en.wikipedia.org/wiki/Box_blur).", "Box Blur")
    }
    if blur_type in blur_info:
        st.markdown(f"### {blur_info[blur_type][1]}")
        st.markdown(blur_info[blur_type][0])
    elif blur_type == 'sharp':
        st.markdown("### Sharp Image")
        st.markdown("Your image is sharp and clear. No blur detected!")

st.title('PureView: Image Enhancement and Deblurring')

enhancement_type = st.sidebar.selectbox('Enhancement Type', ['contrast', 'sharpness', 'brightness'])
enhancement_factor = st.sidebar.slider('Enhancement Factor', 0.5, 2.0, 1.1, 0.1)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    blur_type = classify_image(original_image)

    # Deblur and enhance the image, ensuring it matches the original size
    enhanced_image_resized = deblur_and_enhance_image(original_image, blur_type, enhancement_type, enhancement_factor)

    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.image(original_image, caption='Uploaded Image.', width=max_width)  # Use fixed width

    with col2:
        st.write(f"Image classified as {blur_type}. Processed:")
        st.image(enhanced_image_resized, caption='Enhanced Image.', width=max_width)
        display_blur_info(blur_type)
    
    # Prepare the enhanced image for download
    buf = io.BytesIO()
    enhanced_image_resized.save(buf, format="JPEG")
    st.download_button(label="Download Enhanced Image", data=buf.getvalue(), file_name="enhanced_image.jpg", mime="image/jpeg")
