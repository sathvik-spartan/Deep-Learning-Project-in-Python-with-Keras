import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import keras_tuner as kt  

# Load and preprocess data
def load_data(filepath: str = "pima-indians-diabetes.csv"):
    dataset = np.loadtxt(filepath, delimiter=",")
    X = dataset[:, 0:8]
    y = dataset[:, 8]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# Build model for KerasTuner
def build_model(hp, input_dim=8):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_1', 64, 256, step=64),
                    input_shape=(input_dim,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(units=hp.Int('units_2', 32, 128, step=32),
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# K-Fold Cross-validation
def cross_validate_model(X, y, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Sequential()
        model.add(Dense(128, input_shape=(8,), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        history = model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0,
                            validation_split=0.2, callbacks=[early_stopping, lr_reduction])

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(accuracy)

        predictions = model.predict(X_test)
        rounded_predictions = [round(x[0]) for x in predictions]

        print("Classification Report:")
        print(classification_report(y_test, rounded_predictions))
        plot_confusion_matrix(y_test, rounded_predictions, title="Fold Confusion Matrix")

    print(f"\nAverage Accuracy: {np.mean(accuracies) * 100:.2f}%")

# Hyperparameter Tuning
def hyperparameter_search(X_train, y_train):
    tuner = kt.Hyperband(
        lambda hp: build_model(hp),
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory='ktuner_dir',
        project_name='diabetes_tuning'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]

    print("\nBest Hyperparameters:")
    print(best_hp.values)

    return best_model

# Main
def main():
    X, y = load_data()

    print("Running K-Fold Cross-Validation")
    cross_validate_model(X, y)

    print("\nRunning Hyperparameter Tuning")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model = hyperparameter_search(X_train, y_train)

    print("\nEvaluating Best Model on Test Set...")
    history = best_model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
    plot_history(history)

    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy with Best Model: {accuracy * 100:.2f}%")

    predictions = best_model.predict(X_test)
    rounded_predictions = [round(x[0]) for x in predictions]

    print("Classification Report (Best Model):")
    print(classification_report(y_test, rounded_predictions))
    plot_confusion_matrix(y_test, rounded_predictions, title="Best Model Confusion Matrix")

if __name__ == "__main__":
    main()
