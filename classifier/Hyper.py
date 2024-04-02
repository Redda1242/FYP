from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from keras.callbacks import EarlyStopping, CSVLogger
from data_loader import load_images

csv_logger = CSVLogger('training_log26.csv', append=True, separator=';')

#actual
X, y = load_images('testing\dataset_test')

class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape,
                   kernel_regularizer=l2(hp.Float('reg_rate', min_value=1e-5, max_value=1e-2, sampling='LOG'))),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.05)),

            Conv2D(64, (3, 3), activation='relu',
                   kernel_regularizer=l2(hp.Float('reg_rate', min_value=1e-5, max_value=1e-2, sampling='LOG'))),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.05)),

            Conv2D(128, (3, 3), activation='relu',
                   kernel_regularizer=l2(hp.Float('reg_rate', min_value=1e-5, max_value=1e-2, sampling='LOG'))),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.05)),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(hp.Float('dropout_4', min_value=0.0, max_value=0.5, step=0.05)),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

# Instantiate the hypermodel
hypermodel = MyHyperModel(input_shape=(224, 224, 3), num_classes=5)

# Configure the random search tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=2,
    directory='my_dir',
    project_name='keras_tuner_demo'
)

# Start the search for the best hyperparameter configuration
tuner.search(X, y, epochs=10, validation_split=0.2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal regularization rate is {best_hps.get('reg_rate')} 
and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
model.fit(X, y, epochs=10, validation_split=0.2, callbacks=[csv_logger])
