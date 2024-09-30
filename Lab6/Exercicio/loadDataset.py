import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import random
def create_pairs(img, digit_indices):
    """
    This function create pairs of images. It alternates
    between positive pairs and negative pairs.
    """
    # Define list to store pairs and labels
    pairs = []
    labels = []

    # Get the number of pairs possible
    nb_pairs = min([len(digit_indices[d]) for d in range(10)]) - 1

    for d in range(10):
        for i in range(nb_pairs):
            # Create identical pairs (positive pairs)
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[img[z1], img[z2]]]

            # Create negative pairs
            dn = (d + random.randrange(1, 10)) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[img[z1], img[z2]]]

            # Add the labels
            labels += [1, 0]

    return np.array(pairs), np.array(labels)

def get_pairs(images, labels):
    """
    This function get the pairs of images and the
    corresponding labels.
    """
    # Get the index for each digits
    digit_indices = [np.where(labels == i)[0] for i in range(10)]

    pairs, y = create_pairs(images, digit_indices)

    return pairs, y

# Cargar el dataset Fashion MNIST
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalizar las imágenes
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Expandir la dimensión para adaptarse a la entrada de la red
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)
train_pairs, train_pairs_labels = get_pairs(train_images, train_labels)
test_pairs, test_pairs_labels = get_pairs(test_images, test_labels)



# Guardar los datos del Cliente 1 en un archivo npz
np.savez('fashion_mnist_pairs.npz',
         x_train=train_pairs,
         y_train=train_pairs_labels,
         x_test=test_pairs,
         y_test=test_pairs_labels)

print("Datos guardados correctamente en archivos .npz.")