"""AssignmentAI.ipynb - core code excerpt"""

import os, random, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from PIL import Image

# Mount Drive and dataset path (Colab)
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/Dataset/Dataset"

IMG_SIZE = (64,64)
COLOR_MODE = 'rgb'   # or 'grayscale' depending on your images
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset='training', seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode=COLOR_MODE)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset='validation', seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode=COLOR_MODE)

# Preprocessing and augmentation
rescale = layers.Rescaling(1./255)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.05),
])

def prepare(ds, training=False):
    ds = ds.map(lambda x,y: (rescale(x), y))
    if training: ds = ds.map(lambda x,y: (data_augmentation(x, training=True), y))
    return ds

train_ds = prepare(train_ds, training=True)
val_ds   = prepare(val_ds)

# Build model (best hyperparams)
def build_model(input_shape=(64,64,3), num_classes=10, dropout_rate=0.2, l2_lambda=1e-5):
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(l2_lambda),
                            input_shape=input_shape))
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

channels = 3 if COLOR_MODE == 'rgb' else 1
model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], channels),
                    num_classes=len(train_ds.class_names),
                    dropout_rate=0.2, l2_lambda=1e-5)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
    keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=60, callbacks=callbacks)
model.save('final_model.keras')

# Evaluation
y_true, y_pred, y_scores = [], [], []
for images, labels in val_ds:
    probs = model.predict(images)
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)
    y_scores.extend(probs)

print(classification_report(y_true, y_pred, digits=4))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.show()