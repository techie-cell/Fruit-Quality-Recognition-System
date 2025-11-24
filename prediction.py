# ===========================================
# 🔍 STEP 1: Import Required Libraries
# ===========================================
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# ===========================================
# 📁 STEP 2: Define Paths
# ===========================================
model_path = 'vgg16_final_model.h5'  # Path to your saved model
dataset_path = 'train'               # Path to your training folder (to get class names)
img_path = 'test_image.jpg'          # Replace with your image file path

# ===========================================
# 📂 STEP 3: Load Model and Class Labels
# ===========================================
model = load_model(model_path)
print("✅ Model loaded successfully!")

# Get class names from folder structure
class_labels = sorted(os.listdir(dataset_path))
print(f"Loaded {len(class_labels)} classes:")
print(class_labels)

# ===========================================
# 🖼️ STEP 4: Load and Preprocess Image
# ===========================================
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = x / 255.0  # normalize
x = np.expand_dims(x, axis=0)

# ===========================================
# 🧠 STEP 5: Predict
# ===========================================
pred = model.predict(x)
predicted_index = np.argmax(pred)
predicted_class = class_labels[predicted_index]
confidence = np.max(pred) * 100

# ===========================================
# 📊 STEP 6: Display Result
# ===========================================
plt.imshow(image.load_img(img_path))
plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
plt.axis('off')
plt.show()

print(f"🧾 Predicted Class: {predicted_class}")
print(f"🎯 Confidence: {confidence:.2f}%")
