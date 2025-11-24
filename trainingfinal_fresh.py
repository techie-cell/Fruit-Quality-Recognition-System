# ===========================================
# 📦 STEP 1: Import Libraries
# ===========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from PIL import Image

# ===========================================
# 🗂️ STEP 2: Dataset Setup
# ===========================================
base_dir = 'Test'  # your folder containing 18 subfolders

folders = sorted(os.listdir(base_dir))
fruit_names = sorted(list(set([f.replace('fresh', '').replace('rotten', '') for f in folders])))

print("🍎 Fruits detected:", fruit_names)
print("Total folders:", len(folders))

# ===========================================
# 🧾 STEP 3: Create DataFrame for Images
# ===========================================
data = []
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, img_file)
                fruit_type = folder.replace('fresh', '').replace('rotten', '')
                freshness = 'fresh' if 'fresh' in folder else 'rotten'
                data.append([img_path, fruit_type, freshness])

df = pd.DataFrame(data, columns=['filepath', 'fruit', 'freshness'])
print(f"✅ Loaded {len(df)} images")
df.head()

# ===========================================
# 📊 STEP 4: Encode Labels
# ===========================================
from sklearn.preprocessing import LabelEncoder

fruit_encoder = LabelEncoder()
df['fruit_label'] = fruit_encoder.fit_transform(df['fruit'])

fresh_encoder = LabelEncoder()
df['fresh_label'] = fresh_encoder.fit_transform(df['freshness'])  # 0 = rotten, 1 = fresh

num_fruits = len(fruit_encoder.classes_)
print("🍉 Fruit classes:", fruit_encoder.classes_)
print("🌿 Freshness classes:", fresh_encoder.classes_)

# ===========================================
# 🧹 STEP 5: Train/Validation Split
# ===========================================
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['fruit_label'], random_state=42)
print(f"Train: {len(train_df)}, Validation: {len(val_df)}")

# ===========================================
# 📦 STEP 6: Custom Data Generator
# ===========================================
class DualOutputDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size, img_size=(224,224), augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25 if augment else 0,
            zoom_range=0.2 if augment else 0,
            horizontal_flip=augment
        )

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.df.iloc[idx*self.batch_size:(idx+1)*self.batch_size]
        imgs, fruit_labels, fresh_labels = [], [], []

        for _, row in batch.iterrows():
            img = Image.open(row.filepath).convert('RGB').resize(self.img_size)
            img = np.array(img)
            imgs.append(img)
            fruit_labels.append(row.fruit_label)
            fresh_labels.append(row.fresh_label)

        X = np.array(imgs)
        X = self.datagen.standardize(X)
        y_fruit = to_categorical(fruit_labels, num_classes=num_fruits)
        y_fresh = np.array(fresh_labels)
        return X, {'fruit_output': y_fruit, 'fresh_output': y_fresh}

train_gen = DualOutputDataGenerator(train_df, batch_size=32, augment=True)
val_gen = DualOutputDataGenerator(val_df, batch_size=32, augment=False)

# ===========================================
# 👀 STEP 7: Visualize Sample Images
# ===========================================
plt.figure(figsize=(10, 10))
for i in range(9):
    X_batch, y_batch = train_gen[i]
    img = X_batch[0]
    fruit_idx = np.argmax(y_batch['fruit_output'][0])
    fresh_val = y_batch['fresh_output'][0]
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{fruit_encoder.classes_[fruit_idx]} - {'Fresh' if fresh_val==1 else 'Rotten'}")
plt.show()

# ===========================================
# 🧠 STEP 8: Build Multi-Output Model (VGG16)
# ===========================================
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

fruit_output = Dense(num_fruits, activation='softmax', name='fruit_output')(x)
fresh_output = Dense(1, activation='sigmoid', name='fresh_output')(x)

model = Model(inputs=base_model.input, outputs=[fruit_output, fresh_output])
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={'fruit_output': 'categorical_crossentropy', 'fresh_output': 'binary_crossentropy'},
    metrics={'fruit_output': 'accuracy', 'fresh_output': 'accuracy'}
)
model.summary()

# ===========================================
# 🏋️ STEP 9: Train the Model
# ===========================================
checkpoint = ModelCheckpoint('fruit_freshness_best.h5', monitor='val_fruit_output_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_fruit_output_accuracy', patience=5, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    callbacks=[checkpoint, early_stop]
)

# ===========================================
# 📈 STEP 10: Plot Training Graphs
# ===========================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['fruit_output_accuracy'], label='Fruit Acc')
plt.plot(history.history['val_fruit_output_accuracy'], label='Val Fruit Acc')
plt.title('Fruit Classification Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['fresh_output_accuracy'], label='Freshness Acc')
plt.plot(history.history['val_fresh_output_accuracy'], label='Val Freshness Acc')
plt.title('Freshness Accuracy')
plt.legend()
plt.show()

# ===========================================
# 💾 STEP 11: Save Model
# ===========================================
model.save('fruit_freshness_final.h5')
print("✅ Saved as fruit_freshness_final.h5")
