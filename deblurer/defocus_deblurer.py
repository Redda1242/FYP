from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add
from keras.models import Model

def build_enhanced_model():
    input_img = Input(shape=(224, 224, 3)) 

    # Encoder
    # compress input image into lower dimensional representation
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    # reconstruct the input image from the its compressed representation
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Adding residual connection
    residual = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(input_img)
    decoded = Add()([decoded, residual])

    # Model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error' , metrics=['accuracy'])
    
    return autoencoder

