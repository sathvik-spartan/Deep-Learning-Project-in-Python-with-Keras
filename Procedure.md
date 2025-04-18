# Title: Simple Neural Network on Pima Indians Diabetes Dataset
# Author: Based on Machine Learning Mastery by Jason Brownlee
# Description: A basic neural network built using Keras (TensorFlow backend) to perform binary classification on the Pima Indians Diabetes dataset.

# ===============================
### 📦 Step 1: Import Dependencies
# ===============================
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
### 📊 Step 2: Load the Dataset
# ===============================
# Dataset format: 8 features + 1 output (0 or 1)
# Download CSV from: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
# Save it as 'pima-indians-diabetes.csv' in the same folder

print("📥 Loading dataset...")
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]  # Input features
y = dataset[:, 8]    # Labels (binary)

# Optional: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
### 🧠 Step 3: Define the Model
# ===============================
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))  # First hidden layer
model.add(Dense(8, activation='relu'))                      # Second hidden layer
model.add(Dense(1, activation='sigmoid'))                   # Output layer

# ===============================
### ⚙️ Step 4: Compile the Model
# ===============================
model.compile(
    loss='binary_crossentropy',  # Binary classification
    optimizer='adam',            # Optimizer
    metrics=['accuracy']         # Track accuracy
)

# ===============================
### 🚀 Step 5: Train the Model
# ===============================
print("🧪 Training the model...")
model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=1)

# ===============================
### ✅ Step 6: Evaluate the Model
# ===============================
print("\n📊 Evaluating model on test data...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# ===============================
### 🔍 Step 7: Make Predictions
# ===============================
print("\n🔍 Generating predictions...")
predictions = model.predict(X_test)
rounded_predictions = [round(x[0]) for x in predictions]

# Print metrics
print("\n📄 Classification Report:")
print(classification_report(y_test, rounded_predictions))
print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, rounded_predictions))
