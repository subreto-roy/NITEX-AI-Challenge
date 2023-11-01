
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Dip/fashion-mnist_train.csv")


x_train = data.iloc[:, 1:]
y_train = data.iloc[:, 0]

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


#basic information dataset
print(f"Training set shape: {x_train.shape}")
print(f"Testing set shape: {x_test.shape}")
print(f"Number of classes: {len(np.unique(y_train))}")

# Visualize training set
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train, palette="viridis")
plt.title("Distribution of Classes in Training Set")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Display each class
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[y_train == i][0], cmap="gray")
    plt.title(class_names[i])
    plt.axis("off")




sns.countplot(x='label', data=data)
plt.title('Class Distribution')
plt.show()


class_samples = data.groupby('label').first()

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

# Load the data
(train_images, train_labels), (val_images, val_labels) = fashion_mnist.load_data()

# Normalize
train_images, val_images = train_images / 255.0, val_images / 255.0

# Reshape
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
val_images = val_images.reshape((val_images.shape[0], 28, 28, 1))

#CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(val_images, val_labels)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('fashion_model.h5')

import sys
import tensorflow as tf
import numpy as np

def load_and_preprocess_eval_data(dataset_folder):

    eval_data = np.genfromtxt(dataset_folder, delimiter=',', skip_header=1)
    eval_images = eval_data[:, 1:]
    eval_labels = eval_data[:, 0]

    # Normalize
    eval_images = eval_images / 255.0

    # Reshape
    eval_images = eval_images.reshape(-1, 28, 28)

    return eval_images, eval_labels


def evaluate_model(dataset_folder):

    model = tf.keras.models.load_model('fashion_model.h5')
    eval_images, eval_labels = load_and_preprocess_eval_data(dataset_folder)
    loss, accuracy = model.evaluate(eval_images, eval_labels)

    # Generate output.txt
    with open('output.txt', 'w') as f:
        f.write(f'Model Architecture:\n{model.summary()}\n\n')
        f.write(f'Evaluation Metric (Accuracy): {accuracy}\n')
        f.write('Additional insights or observations...\n')


sys.argv = ["evaluate_model.py", "/content/drive/MyDrive/Colab Notebooks/Dip/fashion-mnist_train.csv"]

if len(sys.argv) != 2:
    print("Usage: python evaluate_model.py /content/drive/MyDrive/Colab Notebooks/Dip/fashion-mnist_train.csv")
    sys.exit(1)

dataset_folder = sys.argv[1]

try:

    evaluate_model(dataset_folder)
except Exception as e:
    print(f"Error: {e}")