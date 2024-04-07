import streamlit as st
from ui_components import setup_page, upload_and_display_image, feedback_form, show_homepage

def main():
    setup_page()

    # Initialize or access existing state
    if 'deblur_started' not in st.session_state:
        st.session_state['deblur_started'] = False

    if not st.session_state['deblur_started']:
        # Display the homepage with a button to start deblurring
        show_homepage()
        if st.button("Start Deblurring"):
            st.session_state['deblur_started'] = True
            # This will cause the app to rerun and the next part of the code will execute

    # Once deblurring has started, handle image upload and processing
    if st.session_state['deblur_started']:
        original_image, blur_type = upload_and_display_image()
        if original_image is not None and blur_type is not None:
            feedback_form()

if __name__ == "__main__":
    main()
