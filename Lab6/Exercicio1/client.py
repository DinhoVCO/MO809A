import siamese_pb2 as pb2
import siamese_pb2_grpc as pb2_grpc
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
import grpc
import time
import numpy as np
import random
import multiprocessing

# Desativar gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_Client_data():
    # Cargar los datos
    path_data = f'fashion_mnist_pairs.npz'
    data_cliente = np.load(path_data)

    x_train_client = data_cliente['x_train']
    y_train_client = data_cliente['y_train']
    x_test_client = data_cliente['x_test']
    y_test_client = data_cliente['y_test']
    return x_train_client,y_train_client,x_test_client,y_test_client

def get_activations(model, X, id):
    with tf.GradientTape(persistent=True) as tape:
        activations1 = model(X[:, 0], training=True)
        activations2 = model(X[:, 1], training=True)
    return activations1, activations2 , tape

def create_siamese_branch(input_layer):
    x = layers.Conv2D(32, 3, activation='elu')(input_layer)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='elu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    return Model(inputs=input_layer, outputs=x)

def send_activations_to_server(stub, activations1,activations2, labels, batch_size, client_id):
    activations_list1 = activations1.numpy().flatten()
    activations_list2 = activations2.numpy().flatten()
    client_to_server_msg = pb2.ClientToServer()
    client_to_server_msg.activations1.extend(activations_list1)
    client_to_server_msg.activations2.extend(activations_list2)
    client_to_server_msg.labels.extend(labels.flatten())
    client_to_server_msg.batch_size = batch_size
    client_to_server_msg.client_id = client_id
    
    server_response = stub.SendClientActivations(client_to_server_msg)

    return server_response

def train_step(model, x_batch, y_batch, step, optimizer, epoch, stub,client_id):
    
    activations1,activations2, tape     = get_activations(model, x_batch,client_id)
    flattened_activations1 = tf.reshape(activations1, (activations1.shape[0], -1))
    flattened_activations2 = tf.reshape(activations2, (activations1.shape[0], -1))
    server_response = send_activations_to_server(stub, flattened_activations1,flattened_activations2, y_batch, len(x_batch),client_id)
    activations_grad1 = tf.convert_to_tensor(server_response.gradients1, dtype=tf.float32)
    activations_grad1 = tf.reshape(activations_grad1, activations1.shape)
    activations_grad2 = tf.convert_to_tensor(server_response.gradients2, dtype=tf.float32)
    activations_grad2 = tf.reshape(activations_grad2, activations2.shape)
    client_gradient1 = tape.gradient(
        activations1,
        model.trainable_variables,
        output_gradients=activations_grad1
    )

    client_gradient2 = tape.gradient(
        activations2,
        model.trainable_variables,
        output_gradients=activations_grad2
    )
    
    # Sumar ambos gradientes ya que las dos ramas comparten pesos
    total_grads = [g1 + g2 for g1, g2 in zip(client_gradient1, client_gradient2)]

    loss      = server_response.loss
    optimizer.apply_gradients(zip(total_grads, model.trainable_variables))

    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss}")
        with open('results.csv', 'a') as f:
            f.write(f"{epoch}, {step}, {loss}\n")

#Only for test
def euclidean_distance(embedding_a, embedding_b):
    """Calcula la distancia euclidiana entre dos embeddings."""
    return tf.sqrt(tf.reduce_sum(tf.square(embedding_a - embedding_b), axis=1))

#Only for test
def contrastive_loss(y_true, y_pred):
    """Pérdida contrastiva."""
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean((1 - y_true) * square_pred + y_true * margin_square)


def batch_generator(pairs, labels, batch_size):
    for i in range(0, len(pairs), batch_size):
        yield pairs[i:i + batch_size], labels[i:i + batch_size]

def main(client_id):
    print(f'cargando dataset del cliente {client_id}...')
    train_pairs, train_pairs_labels,test_pairs, test_pairs_labels = load_Client_data()
    print(f'Descarga concluida para el cliente {client_id}...')

    partial_model      = create_siamese_branch(layers.Input(shape=(28, 28, 1)))
    MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 *10
    
    # Configuração da conexão gRPC
    channel = grpc.insecure_channel('localhost:50051',options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
    stub = pb2_grpc.SiameseStub(channel)
    client_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Número de épocas y tamaño de lote
    epochs = 4
    batch_size = 128
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, (batch_pairs, batch_labels) in enumerate(batch_generator(train_pairs, train_pairs_labels, batch_size)):
            train_step(partial_model, batch_pairs, batch_labels, step , client_optimizer, epoch, stub,client_id)
        
        # Evaluación en cada época
        test_embedding_a = partial_model(test_pairs[:, 0], training=False)
        test_embedding_b = partial_model(test_pairs[:, 1], training=False)
        test_preds = euclidean_distance(test_embedding_a, test_embedding_b)
        test_loss = contrastive_loss(test_pairs_labels, test_preds)
        print(f"Test Loss after epoch {epoch + 1}: {test_loss.numpy()}")
        with open('test_results.csv', 'a') as f:
            f.write(f"{epoch}, {test_loss.numpy()}\n")
    
    # Guardar el modelo después del entrenamiento
    save_path = f"siamese_model_client_{client_id}.h5"
    partial_model.save(save_path)
    print(f"Modelo guardado en {save_path}")
    
if __name__ == '__main__':
    # Crear dos procesos para simular dos clientes
    client1 = multiprocessing.Process(target=main, args=(1,))
    #client2 = multiprocessing.Process(target=main, args=(2,))

    # Iniciar los procesos de los clientes
    client1.start()
    #client2.start()

    # Esperar la finalización de los procesos de los clientes
    client1.join()
    #client2.join()
    
