import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import grpc
from concurrent import futures
import tensorflow as tf
from  tensorflow.keras import layers, models
from keras.models import Model
import time
import pandas as pd
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

loss_to_save = []
epoch = 0

class ServerModel(tf.keras.models.Model):
    def __init__(self):
        super(ServerModel, self).__init__()
        self.layer3 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = layers.Flatten() 
        self.layer4 = layers.Dense(128, activation='relu')
        self.tape = None
    
    def call(self, input):
        x = self.layer3(input)
        x = self.flatten(x)
        x = self.layer4(x)
        return x
    
def create_server_model(input_shape):
    return ServerModel()

class SplitLearningService(pb2_grpc.SplitLearningServicer):

    def __init__(self):
        self.server_model = create_server_model((15,15,32))
        self.optimizer    = tf.keras.optimizers.Adam()
        self.metrics      = tf.keras.metrics.SparseCategoricalAccuracy()
        self.last_input = None 

    def SendActivation(self, request, context):
        batch_size = request.batch_size
        # Recebe ativações do cliente
        activations = tf.convert_to_tensor(request.activations, dtype=tf.float32)
        activations = tf.reshape(activations, (batch_size, 15, 15, 32)) 

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(activations)
            # Realiza feedforward em M2
            activations_m2 = self.server_model(activations, training = True)
            # Envia ativações de M2 de volta ao cliente
            
        self.tape = tape
        self.last_input = activations
        response = pb2.ActivationsResponse()
        response.activations.extend(activations_m2.numpy().flatten())
        return response

    def SendGradient(self, request, context):
        batch_size = request.batch_size
        # Recebe gradientes do cliente
        activations_grad = tf.convert_to_tensor(request.gradients, dtype=tf.float32)
        activations_grad = tf.reshape(activations_grad, (batch_size, 128))

        # Atualiza o modelo M2 com o backpropagation usando os gradientes
        with self.tape:
            server_outputs = self.server_model(self.last_input, training=True)
            gradients = self.tape.gradient(
                server_outputs, self.server_model.trainable_variables, output_gradients=activations_grad
            )

        # Aplica os gradientes para atualizar os pesos de M2
        self.optimizer.apply_gradients(zip(gradients, self.server_model.trainable_variables))

        # Agora, calcule os gradientes das ativações de M1 (para enviar de volta ao cliente)
        # Isso permite que o cliente continue o backpropagation em M1
        activation_gradients = self.tape.gradient(
            server_outputs, self.last_input, output_gradients=activations_grad
        )
        response = pb2.GradientsResponse()

        response.gradients.extend(activation_gradients.numpy().flatten())
        # Enviar gradientes das ativações de M1 de volta ao cliente
        return response


        

# Função principal para iniciar o servidor gRPC
def serve():
    global server
    MAX_MESSAGE_LENGTH = 20 * 1024 * 1024 * 10
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
    pb2_grpc.add_SplitLearningServicer_to_server(SplitLearningService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()
    

if __name__ == '__main__':
    serve()
