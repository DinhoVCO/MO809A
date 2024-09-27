import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import RMSprop

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
    y = y.astype(np.int32)

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

    return Model(inputs=input_layer, outputs=x)


def send_activations_to_server(stub, activations, labels, batch_size, client_id):
    activations_list = activations.numpy().flatten()
    #print(f'activa: {activations_list}')
    client_to_server_msg = pb2.ClientToServer()
    client_to_server_msg.activations.extend(activations_list)
    client_to_server_msg.labels.extend(labels.flatten())
    client_to_server_msg.batch_size = batch_size
    client_to_server_msg.client_id = client_id
    print(f'activa: {activations_list}')
    print(f'L: {labels.flatten()}')
    print(f'client: {client_id}')
    
    server_response = stub.SendClientActivations(client_to_server_msg)

    return server_response

def train_step(model, x_batch, y_batch, batch, optimizer, stub):
    
    activations, tape     = get_activations(model, np.expand_dims(x_batch, axis=0))
    print(f"Activaciones obtenidas para la imagen : {activations}")
    print(f"Forma de las activaciones: {activations.shape}")

    flattened_activations = tf.reshape(activations, (activations.shape[0], -1))

    latencia_start  = time.time()
    print("aquiii")
    server_response = send_activations_to_server(stub, flattened_activations, y_batch, len(x_batch), 1)
    latencia_end    = time.time()

    print(f"Received response from server:")
    loss = server_response.loss
    loss_tensor = tf.convert_to_tensor(loss, dtype=tf.float32)  # or dtype=tf.int32

    print(f'loss rec:{loss_tensor}')
    #activations_grad = tf.convert_to_tensor(server_response.gradients, dtype=tf.float32)
    #activations_grad = tf.reshape(activations_grad, activations.shape)

    client_gradient = tape.gradient(
        loss_tensor,
        model.trainable_variables
    )

    #bytes_tx  = flattened_activations.numpy().nbytes
    #bytes_rx  = activations_grad.numpy().nbytes
    latencia  = latencia_end - latencia_start
    loss      = loss_tensor
    acc       = server_response.acc

    print(f"Latencia: {latencia} segundos")
    #print(f"Data Tx: {bytes_tx / 2**20} MB")
    #print(f"Data Rx: {bytes_rx / 2**20} MB")

    optimizer.apply_gradients(zip(client_gradient, model.trainable_variables))
    
    with open('results.csv', 'a') as f:
        f.write(f"{batch}, {loss}, {acc}, {latencia}\n")

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
    client_optimizer   = RMSprop()
    MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 *10
    
    # Configuração da conexão gRPC
    channel = grpc.insecure_channel('localhost:50051',options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
    stub = pb2_grpc.SplitLearningStub(channel)


    for batch in range(1):
        print(f"image {batch}")
        
        X_batch  = train_pairs[batch][0]
        image_with_channel = np.expand_dims(X_batch, axis=-1)
        y_batch  = train_y[batch]

        train_step(partial_model, image_with_channel, y_batch, batch, client_optimizer, stub)
    
    
if __name__ == '__main__':
    main()
