import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Load MNIST and reshape for image processing (28x28x1)
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train[:1].reshape(-1, 28, 28, 1).astype("float32") / 255

# Define the data augmentation pipeline
# These layers are only active during training (fit)
data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(0.1),      # Rotate images by up to 10%
    keras.layers.RandomZoom(0.1),          # Zoom in/out by up to 10%
    keras.layers.RandomTranslation(0.1, 0.1) # Shift horizontally and vertically
])

# Visualize the effect of augmentation on a single image
plt.figure(figsize=(10, 10))
for i in range(9):
    # Apply the augmentation to the same image 9 times
    augmented_image = data_augmentation(x_train, training=True)
    
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0].numpy().astype("float32").reshape(28, 28), cmap='gray')
    plt.title(f"Augmented version {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Example: Incorporating augmentation into a full model
def build_augmented_model():
    model = keras.Sequential([
        # Add the augmentation as the first layer
        keras.layers.Input(shape=(28, 28, 1)),
        data_augmentation,
        
        # Standard CNN architecture
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

augmented_model = build_augmented_model()
augmented_model.summary()