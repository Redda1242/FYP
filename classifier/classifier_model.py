from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam

do1 =  0.1
do2 = 0.15
do3 = 0.2
do4 = 0.25
regularizer = 0.01


def build_model(input_shape=(224, 224, 3), num_classes=5):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(regularizer)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(do1),

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(regularizer)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(do2),

        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(regularizer)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(do3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(do4),
        Dense(num_classes, activation='softmax')  # Adjust num_classes based on your dataset
    ])

    custom_learning_rate = 0.00001
    optimizer = Adam(learning_rate=custom_learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
