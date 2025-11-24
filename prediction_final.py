# ===========================================
# 🔍 STEP 1: Import Required Libraries
# ===========================================
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# ===========================================
# 📁 STEP 2: Define Paths
# ===========================================
model_path = 'vgg16_final_model.h5'   # Path to saved model
dataset_path = 'Test'                # Folder used for class labels
img_path = '2.png'           # Change this to your test image

# ===========================================
# 🧠 STEP 3: Load Model and Class Labels
# ===========================================
model = load_model(model_path)
print("✅ Model loaded successfully!")

# Get class names (sorted alphabetically like training)
class_labels = sorted(os.listdir(dataset_path))
print(f"Loaded {len(class_labels)} classes:")
print(class_labels)

# ===========================================
# 🖼️ STEP 4: Load and Preprocess Image
# ===========================================
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# ===========================================
# 🧮 STEP 5: Prediction
# ===========================================
pred = model.predict(x)
pred_index = np.argmax(pred)
pred_class = class_labels[pred_index]
confidence = np.max(pred) * 100

# ===========================================
# 🍎 STEP 6: Extract Freshness and Fruit Type
# ===========================================
if 'fresh' in pred_class.lower():
    freshness = "Fresh"
elif 'rotten' in pred_class.lower():
    freshness = "Rotten"
else:
    freshness = "Unknown"

# Remove freshness word to isolate fruit type
fruit_name = pred_class.lower().replace('fresh', '').replace('rotten', '').strip()
fruit_name = fruit_name.capitalize()

# ===========================================
# 🎨 STEP 7: Display Result
# ===========================================
plt.imshow(load_img(img_path))
plt.axis('off')
plt.title(f"{fruit_name} ({freshness}) - {confidence:.2f}% Confidence")
plt.show()

# ===========================================
# 📋 STEP 8: Print Results
# ===========================================
print(f"🍌 Predicted Fruit: {fruit_name}")
print(f"🌿 Freshness Level: {freshness}")
print(f"🎯 Confidence: {confidence:.2f}%")
