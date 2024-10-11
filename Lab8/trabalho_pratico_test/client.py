import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import grpc
import time
import numpy as np
import random

# Desativar gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def create_partial_model_m1(input_layer):
    layer1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    layer2 = layers.MaxPooling2D((2, 2))(layer1)
    return Model(inputs=input_layer, outputs=layer2)

def create_partial_model_m3(input_layer):
    layer5 = layers.Dense(64, activation='relu')(input_layer) 
    layer6 = layers.Dense(10, activation='softmax')(layer5) 
    return Model(inputs=input_layer, outputs=layer6)

def get_activations(model, X):
    with tf.GradientTape(persistent=True) as tape:
        activations = model(X)
    return activations, tape

def train_step(model_m1, model_m3, x_batch, y_batch, batch, optimizer, epoch, stub):

    activations_m1, tape_m1    = get_activations(model_m1, x_batch)
    #activations_m1 = model_m1(x_batch)
    flattened_activations_m1 = tf.reshape(activations_m1, (activations_m1.shape[0], -1))
    # Enviar ativações ao servidor para processar M2
    activations_list = flattened_activations_m1.numpy().flatten()
    activation_message = pb2.ActivationsRequest()
    activation_message.activations.extend(activations_list)
    response = stub.SendActivation(activation_message)
    # Receber as ativações processadas de M2
    activations_m2 = tf.convert_to_tensor(response.activations, dtype=tf.float32)
    activations_m2 = tf.reshape(activations_m2, (64, 128)) # -1 faz com que a dimensao seja calculada automaticamente

    # Forward pass no cliente (M3)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(activations_m2)
        predictions = model_m3(activations_m2)
        loss        = tf.keras.losses.sparse_categorical_crossentropy(y_batch, predictions)
        loss        = tf.reduce_mean(loss)

    print('loss:', loss.numpy())
    # Backpropagation no cliente (M3)
    grads_m3 = tape.gradient(loss, model_m3.trainable_variables)
    optimizer.apply_gradients(zip(grads_m3, model_m3.trainable_variables))
    
    # **Aqui, calcular os gradientes das ativações de M2**
    # Esses gradientes são o que devemos enviar ao servidor
    grads_activations_m2 = tape.gradient(loss, activations_m2)
    # Enviar gradientes das ativações de M2 para o servidor processar o backpropagation em M2
    gradient_message = pb2.GradientsRequest()
    gradient_message.gradients.extend(grads_activations_m2.numpy().flatten())
    response2 = stub.SendGradient(gradient_message)
    # Opcional: Receber gradientes para M1 e atualizar pesos no cliente (M1)
    activations_grad = tf.convert_to_tensor(response2.gradients, dtype=tf.float32)
    activations_grad = tf.reshape(activations_grad, (64, 15, 15, 32))

    client_gradient = tape_m1.gradient(
        activations_m1,
        model_m1.trainable_variables,
        output_gradients=activations_grad
    )
    optimizer_m1 = tf.keras.optimizers.Adam()
    optimizer_m1.apply_gradients(zip(client_gradient, model_m1.trainable_variables))

def main():

    # CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test                      = X_train / 255.0, X_test / 255.0    
    
    partial_model_m1     = create_partial_model_m1(layers.Input(shape=(32, 32, 3)))
    partial_model_m3     = create_partial_model_m3(layers.Input(shape=(128,)))

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
        n_batches  = X_train.shape[0]//batch_size
        
        for batch in range(n_batches):
            print(f"Epoch {epoch} - Batch {batch}/{n_batches}")
            
            X_batch  = X_train[batch_size * batch : batch_size * (batch+1)]
            y_batch  = y_train[batch_size * batch : batch_size * (batch+1)]

            train_step(partial_model_m1, partial_model_m3, X_batch, y_batch, batch, client_optimizer, epoch, stub)
        
    
if __name__ == '__main__':
    main()
