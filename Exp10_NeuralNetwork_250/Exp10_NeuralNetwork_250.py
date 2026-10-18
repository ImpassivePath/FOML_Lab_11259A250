import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=1)
model = tf.keras.Sequential([
    tf.keras.Input(shape=(64,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, verbose=0)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", round(acc, 3))
new_image = X_test[0:1]
prediction = model.predict(new_image, verbose=0)
digit = np.argmax(prediction)
print("Predicted digit:", digit)
