import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image

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

original_img = img_array.astype("uint8")

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = img_array / 255.0

prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    predicted_class = "Malignant"
    confidence = prediction
else:
    predicted_class = "Benign"
    confidence = 1 - prediction

print(f"\nPrediction : {predicted_class}")
print(f"Confidence : {confidence:.4f}")

last_conv_layer_name = "top_conv"

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer(last_conv_layer_name).output,
        model.output
    ]
)

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(img_array)

    loss = predictions[:,0]

grads = tape.gradient(
    loss,
    conv_outputs
)

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0,1,2)
)

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(
    heatmap,
    0
)

heatmap = heatmap / tf.reduce_max(heatmap)

heatmap = heatmap.numpy()

heatmap = cv2.resize(
    heatmap,
    (IMG_SIZE, IMG_SIZE)
)

heatmap = np.uint8(
    255 * heatmap
)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

superimposed_img = cv2.addWeighted(
    original_img,
    0.6,
    heatmap,
    0.4,
    0
)

os.makedirs(
    "../outputs/gradcam",
    exist_ok=True
)

overlay_path = "../outputs/gradcam/gradcam_overlay.png"

cv2.imwrite(
    overlay_path,
    cv2.cvtColor(
        superimposed_img,
        cv2.COLOR_RGB2BGR
    )
)

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(original_img)
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(
    heatmap,
    cv2.COLOR_BGR2RGB
))
plt.title("Heatmap")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(superimposed_img)
plt.title(predicted_class)
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "../outputs/gradcam/final_visualization.png"
)

plt.show()