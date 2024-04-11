from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping, CSVLogger
from data_loader import load_images
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras_tuner import HyperModel, RandomSearch
csv_logger = CSVLogger('training_log27_big_dataset.csv', append=True, separator=';')

#actual
X, y = load_images('testing/dataset_test')



class AdvancedHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential([
            # First convolution block
            Conv2D(filters=hp.Int('conv_1_filters', min_value=32, max_value=128, step=32), 
                   kernel_size=(3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            BatchNormalization(),
            Conv2D(filters=hp.Int('conv_2_filters', min_value=32, max_value=128, step=32), 
                   kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)),
            
            # Second convolution block
            Conv2D(filters=hp.Int('conv_3_filters', min_value=64, max_value=256, step=64), 
                   kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=hp.Int('conv_4_filters', min_value=64, max_value=256, step=64), 
                   kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(rate=hp.Float('dropout_2', min_value=0.3, max_value=0.6, step=0.1)),
            
            # Third convolution block
            Conv2D(filters=hp.Int('conv_5_filters', min_value=128, max_value=512, step=128), 
                   kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=hp.Int('conv_6_filters', min_value=128, max_value=512, step=128), 
                   kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(rate=hp.Float('dropout_3', min_value=0.4, max_value=0.7, step=0.1)),
            
            GlobalAveragePooling2D(),
            Dense(units=hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'),
            Dropout(rate=hp.Float('dropout_4', min_value=0.5, max_value=0.8, step=0.1)),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

input_shape = (224, 224, 3)
num_classes = 5

hypermodel = AdvancedHyperModel(input_shape=input_shape, num_classes=num_classes)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='hyperparam_tuning',
    project_name='advanced_blur_classifier'
)

tuner.search_space_summary()

# Assuming you have loaded your data into X_train, y_train, X_val, y_val
# This will start the hyperparameter search process
tuner.search(X, y, epochs=10, validation_split=0.2)

# Get the best model
model = tuner.get_best_models(num_models=1)[0]
model.summary()

# Optionally, you can continue training the best model found or directly use it for predictions.

# Build the model with the optimal hyperparameters and train it on the data
#model = tuner.hypermodel.build(best_hps)
model.fit(X, y, epochs=10, validation_split=0.2, callbacks=[csv_logger])
model.save('model_trained/classifier.keras')