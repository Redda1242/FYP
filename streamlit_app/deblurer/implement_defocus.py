import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def deblur_image(image_path, model_path='C:/Users/aljum/OneDrive/Desktop/PureViewApp/model_trained/defocus.keras', target_size=(224, 224)):
    # Load the pre-trained model
    model = load_model(model_path)

    # Load and preprocess the image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1] if your model expects this range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    save_path = 'C:/Users/aljum/OneDrive/Desktop/PureViewApp/deblurer/deblurred_img'
    # Predict the deblurred image
    deblurred = model.predict(img_array)

    # Postprocess the deblurred image
    deblurred = np.clip(deblurred.squeeze(), 0, 1)  # Remove batch dimension and clip values

    # Display the original and deblurred images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(deblurred)
    plt.title('Deblurred Image')
    plt.axis('off')

    # plt.show()
        # Convert deblurred image back to PIL Image and save
    deblurred_image = Image.fromarray((deblurred * 255).astype(np.uint8))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_filename = os.path.basename(image_path)
    save_filename = os.path.join(save_path, f"deblurred_{image_filename}")
    deblurred_image.save(save_filename)
    print(f"Saved deblurred image to {save_filename}")

            
deblur_image('C:/Users/aljum/OneDrive/Desktop/PureViewApp/dataset/defocused_blurred/2_XIAOMI-PROCOFONE-F1_F.jpg')
