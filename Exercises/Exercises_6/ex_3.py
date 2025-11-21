import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, AdamW, Adamax, Adadelta, Adagrad

# --- Daten laden ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255

# --- Modellfunktion ---
def build_model():
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# --- Optimizer testen ---
optimizers = {
    "Adam": Adam(),
    "AdamW": AdamW(),
    "Adamax": Adamax(),
    "Adagrad": Adagrad(),
    "Adadelta": Adadelta()
}

results = {}

for name, opt in optimizers.items():
    print(f"\nTraining with optimizer: {name}")
    model = build_model()
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=128,
        verbose=0  # unterdrückt die Ausgabe
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results[name] = test_acc
    print(f"Final accuracy: {test_acc:.4f}")

# --- Ergebnisse ausgeben ---
print("\n===== Vergleich der Optimizer (Final Accuracy) =====")
for k, v in results.items():
    print(f"{k:10s} -> {v:.4f}")
