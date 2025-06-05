import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split

# Load class names dynamically from the dataset directory
class_names = sorted(os.listdir("landmarks_dataset"))
num_classes = len(class_names)

# Prepare the dataset (You need to load the landmarks data from your dataset here)
def prepare_data():
    data = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join("landmarks_dataset", class_name)
        for filename in os.listdir(class_dir):
            # Assuming landmarks are saved as npy files (or you may use other formats)
            if filename.endswith('.npy'):
                landmarks = np.load(os.path.join(class_dir, filename))
                data.append(landmarks)
                labels.append(class_names.index(class_name))
    return np.array(data), np.array(labels)

# Prepare the dataset
data, labels = prepare_data()

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("sign_landmark_model.h5")

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# --- Step 1: Accuracy and Loss Graphs ---

# Plot training & validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot training & validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# --- Step 2: Confusion Matrix ---

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Identify misclassifications
misclassifications = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i, j] > 0:
            misclassifications.append(f"True class '{class_names[i]}' predicted as '{class_names[j]}' {cm[i, j]} times")

# Print misclassifications
if misclassifications:
    print("Misclassifications:")
    for misclass in misclassifications:
        print(misclass)
else:
    print("No misclassifications found.")

# --- Step 3: Classification Report ---

# Classification Report
cr = classification_report(y_test, y_pred_classes, target_names=class_names)

# Print the classification report in the terminal
print("Classification Report:")
print(cr)

# Plot Classification Report as a heatmap
cr_dict = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)
cr_df = pd.DataFrame(cr_dict).transpose()

plt.figure(figsize=(8, 6))
sns.heatmap(cr_df.iloc[:-1, :].astype(float), annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.show()
