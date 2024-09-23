import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist


from tensorflow.keras.models import Model
import grpc
import time
import numpy as np
import random

# Desativar gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Util functions (include preprocessing)
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
    y = y.astype(np.float32)

    return pairs, y

def get_activations(model, X):
    with tf.GradientTape(persistent=True) as tape:
        activations = model(X)
    return activations, tape

def create_partial_model(input_layer):
    """
    This function create the base network of the Siamese
    Network using CNN. It returns a model (note that it
    uses the Functional API of Tensorflow).
    """
    #input_ = layers.Input(shape=(28, 28, 1), name="base_input")

    x = layers.Conv2D(32, 3, activation='elu')(input_layer)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, 3, activation='elu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)

    return Model(inputs=input_, outputs=x)

"""
def create_partial_model(input_layer):
    layer1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    layer2 = layers.MaxPooling2D((2, 2))(layer1)
    layer3 = layers.Conv2D(64, (3, 3), activation='relu')(layer2)
    layer4 = layers.Dense(128, activation='relu')(layer3)
    return Model(inputs=input_layer, outputs=layer4)
"""

def send_activations_to_server(stub, activations, labels, batch_size, client_id):
    activations_list = activations.numpy().flatten()

    client_to_server_msg = pb2.ClientToServer()
    client_to_server_msg.activations.extend(activations_list)
    client_to_server_msg.labels.extend(labels.flatten())
    client_to_server_msg.batch_size = batch_size
    client_to_server_msg.client_id = client_id
    
    server_response = stub.SendClientActivations(client_to_server_msg)

    return server_response

def train_step(model, x_batch, y_batch, batch, optimizer, epoch, stub):
    
    activations, tape     = get_activations(model, x_batch)
    flattened_activations = tf.reshape(activations, (activations.shape[0], -1))

    latencia_start  = time.time()
    server_response = send_activations_to_server(stub, flattened_activations, y_batch, len(x_batch), 1)
    latencia_end    = time.time()

    print("Received response from server")
    activations_grad = tf.convert_to_tensor(server_response.gradients, dtype=tf.float32)
    activations_grad = tf.reshape(activations_grad, activations.shape)

    client_gradient = tape.gradient(
        activations,
        model.trainable_variables,
        output_gradients=activations_grad
    )

    bytes_tx  = flattened_activations.numpy().nbytes
    bytes_rx  = activations_grad.numpy().nbytes
    latencia  = latencia_end - latencia_start
    loss      = server_response.loss
    acc       = server_response.acc

    print(f"Latencia: {latencia} segundos")
    print(f"Data Tx: {bytes_tx / 2**20} MB")
    print(f"Data Rx: {bytes_rx / 2**20} MB")

    optimizer.apply_gradients(zip(client_gradient, model.trainable_variables))
    
    with open('results.csv', 'a') as f:
        f.write(f"{epoch}, {batch}, {loss}, {acc}, {latencia}, {bytes_tx / 2**20}, {bytes_rx / 2**20}\n")

def main():

    # Load the data using the keras dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # Preprocessing
    train_images = train_images.astype(np.float32)
    train_images = train_images / 255.0

    test_images = test_images.astype(np.float32)
    test_images = test_images / 255.0

    # Create pairs based on FashionMNIST
    train_pairs, train_y = get_pairs(train_images, train_labels)
    test_pairs, test_y = get_pairs(test_images, test_labels)

    partial_model      = create_partial_model(layers.Input(shape=(28, 28, 1)))
    client_optimizer   = tf.keras.optimizers.Adam()
    MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 *10
    
    # Configuração da conexão gRPC
    channel = grpc.insecure_channel('localhost:50051',options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
    stub = pb2_grpc.SplitLearningStub(channel)

    for epoch in range(10):
        batch_size = 64
        n_batches  = train_images.shape[0]//batch_size
        
        for batch in range(n_batches):
            print(f"Epoch {epoch} - Batch {batch}/{n_batches}")
            
            X_batch  = train_images[batch_size * batch : batch_size * (batch+1)]
            y_batch  = train_images[batch_size * batch : batch_size * (batch+1)]

            train_step(partial_model, X_batch, y_batch, batch, client_optimizer, epoch, stub)
        
    
if __name__ == '__main__':
    main()
