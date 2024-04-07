import sqlite3
import time
import streamlit as st
from image_processing import classify_image, deblur_and_enhance_image, blur_tips
from PIL import Image
import io

from database import add_feedback_to_db

def setup_page():
    st.set_page_config(page_title="PureView - Image Processing App", page_icon=":camera:", layout="wide")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .feedback-form {
        padding: 10px;
        background-color: #f1f3f6;
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    </style>
    """, unsafe_allow_html=True)
    

def upload_and_display_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        blur_type = classify_image(original_image)
        enhancement_controls(original_image, blur_type)
        return original_image, blur_type
    return None, None

def enhancement_controls(original_image, blur_type):
    st.sidebar.header("Enhancement Controls")
    enhancement_type = st.sidebar.selectbox('Enhancement Type', ['contrast', 'sharpness', 'brightness'])
    enhancement_factor = st.sidebar.slider('Enhancement Factor', 0.5, 2.0, 1.1)
    enhanced_image_resized = deblur_and_enhance_image(original_image, blur_type, enhancement_type, enhancement_factor)
    display_images(original_image, enhanced_image_resized, blur_type)

def display_images(original_image, enhanced_image, blur_type):
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image(original_image, caption='Uploaded Image', width=400, use_column_width=True)
    with col2:
        st.image(enhanced_image, caption='Enhanced Image', width=400, use_column_width=True)
    st.subheader(f"Image classified as {blur_type}")
    st.write(blur_tips(blur_type))
    download_button(enhanced_image)

def download_button(enhanced_image):
    buf = io.BytesIO()
    enhanced_image.save(buf, format="JPEG")
    buf.seek(0)
    st.download_button(
        label="Download Enhanced Image",
        data=buf,
        file_name="enhanced_image.jpg",
        mime="image/jpeg"
    )

def feedback_form():
    image_id = str(int(time.time()))
    with st.form(key='feedback_form', clear_on_submit=True):
        st.markdown('<div class="feedback-form">', unsafe_allow_html=True)
        classification_correct = st.radio("Was the image classification correct?", ("Yes", "No"), horizontal=True)
        deblur_satisfaction = st.slider("How satisfied are you with the deblurring?", 1, 5, 3)
        additional_comments = st.text_area("Any additional comments?")
        submit_feedback = st.form_submit_button('Submit Feedback')

        if submit_feedback:
            conn = sqlite3.connect('feedback.db')  # Ensure this is the correct path to your database
            try:
                classification_correct_bool = classification_correct == "Yes"
                add_feedback_to_db(conn, image_id, classification_correct_bool, deblur_satisfaction, additional_comments)
                conn.commit()  # Ensure commit is called to save the transaction
                st.success('Thank you for your feedback!')
            except Exception as e:
                st.error('Failed to submit feedback. Please try again.')
                print("Error submitting feedback:", e)
            finally:
                conn.close()

        st.markdown('</div>', unsafe_allow_html=True)


def show_homepage():
    """Displays the homepage content."""
    st.title('Welcome to PureView!')
    st.header('Your go-to app for image deblurring and enhancement.')

    st.markdown("""
    PureView is designed to help photographers and photography enthusiasts improve their images. 
    Whether you're dealing with motion blur, defocus, or just want to enhance the overall sharpness and contrast, PureView has got you covered.
    
    **Features include:**
    - **Upload and Deblur:** Automatically detect and correct various types of blur in your images.
    - **Further Enhancements:** Adjust contrast, sharpness, and brightness to give your images that professional look.
    - **Educational Section:** Learn about different types of blur and how to prevent them, enhancing your photography skills.
    
    Get started by choosing an option from the sidebar. Happy enhancing!
    """)

    st.image('https://via.placeholder.com/800x400?text=PureView+Example', caption='Example Image Processed by PureView')