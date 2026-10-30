import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load MNIST and scale pixel values from 0-255 to 0-1
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# 2. Build a small CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),   # find simple patterns (edges)
    layers.MaxPooling2D((2, 2)),                    # shrink the image
    layers.Conv2D(64, (3, 3), activation="relu"),   # find bigger shapes (curves, loops)
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),                               # turn 2D features into a 1D list
    layers.Dense(128, activation="relu"),           # combine features
    layers.Dense(10, activation="softmax"),         # 10 probabilities, one per digit
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 3. Train
history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test),
)

# 4. Evaluate and save
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
model.save("mnist_cnn_model.keras")
print("Model saved to mnist_cnn_model.keras")

# 5. Plot training curves
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("training_curves.png")
