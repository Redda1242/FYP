from images_loader import load_images
from defocus_deblurer import build_enhanced_model
from keras.callbacks import EarlyStopping

# Load your datasets
x_train = load_images('dataset/box_blurred')  
y_train = load_images('dataset/sharp')

# Build the model
autoencoder = build_enhanced_model()
    
# Configure early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

# Train the model with a validation split
autoencoder.fit(x_train, y_train,
                epochs=500,  # Number of epochs to train for
                batch_size=16,  # Batch size
                shuffle=True,  # Shuffle training data
                validation_split=0.2,  # Use 20% of the data for validation
                callbacks=[early_stopping])  # Include early stopping

autoencoder.save('model_trained/box_blurred.keras')

