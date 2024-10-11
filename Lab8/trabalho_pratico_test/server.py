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

        # Recebe ativações do cliente
        activations = tf.convert_to_tensor(request.activations, dtype=tf.float32)
        print(f'Activations shape: {activations.shape}')
        activations = tf.reshape(activations, (64, 15, 15, 32)) 
        print(f'reshape activations shape: {activations.shape}')
        self.last_input = activations
        # Realiza feedforward em M2
        activations_m2 = self.server_model(activations)
        print(f'Activations m2 shape: {activations_m2.shape}')
        # Envia ativações de M2 de volta ao cliente
        response = pb2.ActivationsResponse()

        print('test')
        response.activations.extend(activations_m2.numpy().flatten())
        print('test2')
        return response

    def SendGradient(self, request, context):
        # Recebe gradientes do cliente
        gradients_m3 = tf.convert_to_tensor(request.gradients, dtype=tf.float32)
        gradients_m3 = tf.reshape(gradients_m3, (64, 128))

        # Atualiza o modelo M2 com o backpropagation usando os gradientes
        with tf.GradientTape() as tape:
            # Calcula as ativações em M2
            activations_m2 = self.server_model(self.last_input)
            loss = tf.reduce_mean(tf.square(gradients_m3 - activations_m2))
        
        # Calcular gradientes e atualizar pesos
        grads_m2 = tape.gradient(loss, self.server_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads_m2, self.server_model.trainable_variables))
        
        # Enviar gradientes para M1 de volta ao cliente
        response = pb2.GradientsResponse(gradients=grads_m2[0].numpy().flatten().tolist())
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
