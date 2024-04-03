import streamlit as st

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

# Call the function to display the homepage
show_homepage()
