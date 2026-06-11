import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

model = tf.keras.models.load_model(
    "../models/breast_cancer_model.keras"
)

img_path = input(
    "\nPaste image path here:\n"
)

img_path = img_path.strip('"')

if not os.path.exists(img_path):
    print("\nImage path not found!")
    exit()

IMG_SIZE = 224

img = image.load_img(
    img_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = img_array / 255.0

prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print("\nPrediction: MALIGNANT")
    print(f"Confidence: {prediction:.4f}")
else:
    print("\nPrediction: BENIGN")
    print(f"Confidence: {1-prediction:.4f}")