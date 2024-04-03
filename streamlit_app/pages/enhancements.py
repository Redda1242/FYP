import io
import streamlit as st
from PIL import Image
from model_utils import enhance_image  # Make sure this is implemented in model_utils.py

def display_image_enhancements():
    """Displays UI elements for applying image enhancements."""
    # Check if there is an image to enhance
    if 'enhanced_image' in st.session_state:
        # Retrieve the image from session state
        image = st.session_state['enhanced_image']
        
        # Display the current enhanced image
        st.image(image, caption='Current Enhanced Image', use_column_width=True)

        # UI for selecting enhancement type and factor
        enhancement_type = st.selectbox('Select Enhancement Type', ['contrast', 'sharpness', 'brightness'])
        enhancement_factor = st.slider('Enhancement Factor', 0.5, 2.0, 1.1)

        # Button to apply the selected enhancement
        if st.button('Apply Enhancement'):
            # Apply the enhancement and update the session state
            st.session_state['enhanced_image'] = enhance_image(image, enhancement_type, enhancement_factor)
            st.experimental_rerun()  # Rerun the script to refresh the image display

        # Optional: Button to download the enhanced image
        if st.button('Download Enhanced Image'):
            # Convert the PIL image to bytes and offer it as a download
            buf = io.BytesIO()
            st.session_state['enhanced_image'].save(buf, format='JPEG')
            st.download_button(label="Download Image", data=buf.getvalue(), file_name="enhanced_image.jpg", mime="image/jpeg")

    else:
        st.write("No image found. Please go back and upload an image.")

st.title('Further Enhance Image')
display_image_enhancements()
