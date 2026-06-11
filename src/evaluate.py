import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

model = tf.keras.models.load_model(
    "../models/breast_cancer_model.keras"
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_data = test_datagen.flow_from_directory(
    "../dataset/test",
    target_size=(224,224),
    batch_size=8,
    class_mode='binary',
    shuffle=False
)

predictions = model.predict(test_data)

predicted_classes = (
    predictions > 0.5
).astype(int)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        test_data.classes,
        predicted_classes
    )
)

print("\nClassification Report:\n")

print(
    classification_report(
        test_data.classes,
        predicted_classes
    )
)
