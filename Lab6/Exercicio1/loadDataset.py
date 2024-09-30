import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

def create_pairs(images, labels):
    pairs = []
    labels_pairs = []

    num_classes = len(np.unique(labels))
    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    for idx in range(len(images)):
        current_image = images[idx]
        current_label = labels[idx]

        # Par positivo
        positive_idx = np.random.choice(digit_indices[current_label])
        positive_image = images[positive_idx]

        # Par negativo
        negative_label = (current_label + np.random.randint(1, num_classes)) % num_classes
        negative_idx = np.random.choice(digit_indices[negative_label])
        negative_image = images[negative_idx]

        pairs += [[current_image, positive_image]]
        labels_pairs += [0]

        pairs += [[current_image, negative_image]]
        labels_pairs += [1]

    return np.array(pairs), np.array(labels_pairs)


# Cargar el dataset Fashion MNIST
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalizar las imágenes
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Expandir la dimensión para adaptarse a la entrada de la red
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)
train_pairs, train_pairs_labels = create_pairs(train_images, train_labels)
test_pairs, test_pairs_labels = create_pairs(test_images, test_labels)



# Guardar los datos del Cliente 1 en un archivo npz
np.savez('fashion_mnist_pairs.npz',
         x_train=train_pairs,
         y_train=train_pairs_labels,
         x_test=test_pairs,
         y_test=test_pairs_labels)

print("Datos guardados correctamente en archivos .npz.")