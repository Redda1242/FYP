import streamlit as st

def display_blur_info():
    """Displays educational content on blur types."""
    st.title('Understanding Blur Types')

    # Information about different blur types
    blur_info = {
        'defocus': {
            'description': "Defocus blur occurs when the camera is out of focus. This can happen when the subject moves out of the focus area or when the camera fails to focus correctly.",
            'tips': "To prevent this, ensure your camera's autofocus is on or manually focus on the subject. Use a smaller aperture (higher f-number) for a deeper depth of field, making more of your scene in focus.",
            'link': "https://photographylife.com/what-is-defocus"
        },
        'motion': {
            'description': "Motion blur results from the movement of the subject or camera during exposure. It can create a sense of speed and motion but can also blur details.",
            'tips': "Use a faster shutter speed to freeze motion. If photographing in low light, increase the ISO setting or use flash to allow for faster shutter speeds.",
            'link': "https://www.capturelandscapes.com/how-to-avoid-motion-blur/"
        },
        'camera_shake_blurred': {
            'description': "Camera shake blur happens due to the camera moving while the shutter is open, usually from hand-holding the camera at slow shutter speeds.",
            'tips': "Use a tripod or a monopod to stabilize the camera. Alternatively, use a faster shutter speed or image stabilization features if your camera or lens has them.",
            'link': "https://digital-photography-school.com/how-to-avoid-camera-shake/"
        },
        'box_blurred': {
            'description': "Box blur is a type of blur used in post-processing to smooth out images. It's not caused by camera or subject movement but rather is applied digitally.",
            'tips': "Box blur can be used creatively in post-processing to soften images or backgrounds. It's applied through photo editing software.",
            'link': "https://en.wikipedia.org/wiki/Box_blur"
        }
    }

    # Display the information
    for blur_type, info in blur_info.items():
        st.subheader(blur_type.replace('_', ' ').title())
        st.write(info['description'])
        st.write("Tips: ", info['tips'])
        st.markdown(f"[Learn More]({info['link']})")

# Call the function to display the blur info
display_blur_info()
