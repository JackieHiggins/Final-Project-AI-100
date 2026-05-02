import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Load the dataset (Fashion MNIST)
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

# ---> BUG #12: Casting the images to strings
# This intentionally breaks the model because mathematical layers like 
# Conv2D cannot perform calculations on text-based data types.
train_images = train_images.astype(str)

# 2. Define the Deep Learning Model
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'), 
    layers.Dense(10, activation='softmax') 
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
# This is where the hard crash will occur.
print("Starting training...")
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

# 5. Evaluate the results
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")