import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import grpc
from concurrent import futures
import tensorflow as tf
from  tensorflow.keras import layers, models
import time
import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import threading


def create_server_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = layers.Flatten()(x)
    outputs = layers.Dense(128, activation='relu')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class SplitLearningService(pb2_grpc.SplitLearningServicer):

    def __init__(self):
        self.server_model = create_server_model((15,15,32))
        self.optimizer    = tf.keras.optimizers.Adam()
        self.client_states = {}  # Dictionary to store per-client state
        self.optimizer_lock = threading.Lock()  

    def SendActivation(self, request, context):
        batch_size = request.batch_size
        client_id = request.client_id 
        # Recebe ativações do cliente
        activations = tf.convert_to_tensor(request.activations, dtype=tf.float32)
        activations = tf.reshape(activations, (batch_size, 15, 15, 32)) 

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(activations)
            # Realiza feedforward em M2
            activations_m2 = self.server_model(activations, training = True)
            # Envia ativações de M2 de volta ao cliente
            
        self.client_states[client_id] = {
            'tape': tape, 
            'last_input': activations, 
            'server_outputs': activations_m2
        }
        response = pb2.ActivationsResponse()
        response.activations.extend(activations_m2.numpy().flatten())
        return response

    def SendGradient(self, request, context):
        batch_size = request.batch_size
        client_id = request.client_id 
        client_state = self.client_states.get(client_id)
        if client_state is None:
            context.set_details('Client state not found.')
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return pb2.GradientsResponse()
        tape = client_state['tape']
        last_input = client_state['last_input']
        server_outputs = client_state['server_outputs']

        # Recebe gradientes do cliente
        activations_grad = tf.convert_to_tensor(request.gradients, dtype=tf.float32)
        activations_grad = tf.reshape(activations_grad, (batch_size, 128))
        # Atualiza o modelo M2 com o backpropagation usando os gradientes
        
        gradients = tape.gradient(
            server_outputs, self.server_model.trainable_variables, output_gradients=activations_grad
        )
        # Update the server model weights in a thread-safe manner
        with self.optimizer_lock:
            # Aplica os gradientes para atualizar os pesos de M2
            self.optimizer.apply_gradients(zip(gradients, self.server_model.trainable_variables))



        # Agora, calcule os gradientes das ativações de M1 (para enviar de volta ao cliente)
        # Isso permite que o cliente continue o backpropagation em M1
        activation_gradients = tape.gradient(
            server_outputs, last_input, output_gradients=activations_grad
        )
        # Clean up the client's state
        del self.client_states[client_id]
        del tape 

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
