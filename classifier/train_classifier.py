from keras.models import load_model
from classify_predict import predict_image_class
from classifier_model import build_model
from data_loader import load_images
from keras.callbacks import EarlyStopping, CSVLogger

csv_logger = CSVLogger('training_log25.csv', append=True, separator=';')

#actual
X, y = load_images('testing\dataset_test')
#test
#Xtest, ytest = load_images('C:/Users/aljum/OneDrive/Desktop/PureViewApp/dataset_test')

model = build_model(num_classes=5)  # Adjust based on your dataset

# compile model
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#actual
model.fit(X, y, epochs=15, validation_split=0.2, callbacks=[csv_logger])
model.save('model_trained/classifier.keras')

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


#model.fit(X, y, epochs=50, validation_split=0.2, callbacks=[early_stopping])
#model.save('model_trained/classifier.keras')

#test_loss, test_acc = model.evaluate(X, y, verbose=2)

#test
#model.fit(Xtest, ytest, epochs=10, validation_split=0.2,callbacks=[early_stopping] )
#model.save('model_trained/classifer_test.keras')
#test_loss, test_acc = model.evaluate(Xtest, ytest, verbose=2)

#print('\nTest accuracy:', test_acc)
#print('\nTest Loss:', test_loss)

classifier_model = load_model('model_trained/classifier.keras')
# classifying image
img_path = 'dataset/sharp/134_NIKON-D3400-18-55MM_S.JPG'
predicted_class_name = predict_image_class(img_path)
print("Predicted class:", predicted_class_name)

