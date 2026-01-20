import os
import numpy as np
import tensorflow as tf
import keras

# Load a small subset of MNIST for quick testing
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test[:10].reshape(-1, 784).astype("float32") / 255

# Define and build a simple model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Get original predictions before saving
original_predictions = model.predict(x_test)

# Save the model to disk
# The .keras extension is the recommended native Keras format
model_path = "my_mnist_model.keras"
model.save(model_path)
print(f"Model saved to {model_path}")

# Load the model back
reloaded_model = keras.models.load_model(model_path)
print("Model reloaded successfully")

# Get predictions from the reloaded model
reloaded_predictions = reloaded_model.predict(x_test)

# Check if the results are exactly the same
differences = np.abs(original_predictions - reloaded_predictions)
is_identical = np.allclose(original_predictions, reloaded_predictions)

print(f"Are the predictions identical? {is_identical}")
print(f"Maximum difference between weights/outputs: {np.max(differences)}")

# Cleanup: remove the file from disk
if os.path.exists(model_path):
    os.remove(model_path)