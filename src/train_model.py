import os
import numpy as np
from sklearn.metrics import classification_report
from src.Preprocessing import load_and_preprocess_data
from pipeline import build_model

# Load Preprocessed Data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Build Model
model = build_model(input_shape=(100,100,3))

# Train Model
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes))

# Save Model
if not os.path.exists("model"):
    os.makedirs("model")

model.save("model/mask_detector.h5")

print("Model saved successfully!")