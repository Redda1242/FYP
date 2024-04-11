from tensorflow import keras
from keras import layers, regularizers
from keras_tuner.tuners import Hyperband
from images_loader import load_images
from defocus_deblurer import build_enhanced_model
from tensorflow.keras.callbacks import EarlyStopping

# Load your datasets
x_train = load_images('dataset/defocused_blurred')  
y_train = load_images('dataset/sharp')

def model_builder(hp):
    model = keras.Sequential()
    
    # Assume you are tuning the dropout and regularization in a dense layer
    # Adjust the model architecture as needed
    model.add(layers.InputLayer(input_shape=(224, 224, 3)))  # Example input shape
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    hp_reg = hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='LOG')
    model.add(layers.Dense(units=hp_units, activation='relu', kernel_regularizer=regularizers.l2(hp_reg)))
    model.add(layers.Dropout(hp_dropout))
    model.add(layers.Dense(3, activation='softmax'))  # Adjust the output layer as per your requirement

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

tuner = Hyperband(
    model_builder,
    objective='val_accuracy',  # Adjust this based on your model's metrics
    max_epochs=50,
    factor=3,
    directory='keras_tuner_dir',
    project_name='defocus_tuning'
)

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=best_hps.get('epochs'), validation_split=0.2)

model.save('model_trained/defocus.keras')
