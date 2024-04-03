import streamlit as st
from PIL import Image
import io
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# Import your utility functions from model_utils.py
from model_utils import classify_image, deblur_and_enhance_image, defocus_model, enhance_image, motion_model, box_blur_model, camera_shake_model, preprocess_image

def upload_and_process_image():
    max_width = 350
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        blur_type = classify_image(original_image)

        # Deblur and enhance the image, ensuring it matches the original size
        enhanced_image_resized = deblur_and_enhance_image(original_image, blur_type)

        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            st.image(original_image, caption='Uploaded Image.', width=max_width)  # Use fixed width

        with col2:
            st.write(f"Image classified as {blur_type}. Processed:")
            st.image(enhanced_image_resized, caption='Enhanced Image.', width=max_width)
        
        # Prepare the enhanced image for download
        buf = io.BytesIO()
        enhanced_image_resized.save(buf, format="JPEG")
        st.download_button(label="Download Enhanced Image", data=buf.getvalue(), file_name="enhanced_image.jpg", mime="image/jpeg")


st.title('Upload and Deblur Image')
upload_and_process_image()
