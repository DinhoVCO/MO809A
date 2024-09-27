import splitlearning_pb2 as pb2
import splitlearning_pb2_grpc as pb2_grpc
import grpc
from concurrent import futures
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import time
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

loss_to_save = []
epoch = 0
# Definir la distancia euclidiana (Lambda Layer manual)
def euclidean_distance(embeddings_1, embeddings_2):
    sum_squared = tf.reduce_sum(tf.square(embeddings_1 - embeddings_2))
    return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

# Función de pérdida contrastiva (para la red siamesa)
def contrastive_loss(y_true, y_pred, margin=1):
    # y_true = 1 para pares similares, y_true = 0 para pares diferentes
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)




class SplitLearningService(pb2_grpc.SplitLearningServicer):

    def __init__(self):
        self.metrics      = tf.keras.metrics.SparseCategoricalAccuracy()

    def SendClientActivations(self, request, context):
        print("holaaa")
        activations = tf.convert_to_tensor(request.activations, dtype=tf.float32)
        print(activations.shape)
        #activations = tf.reshape(activations, (request.batch_size, -1)) # -1 faz com que a dimensao seja calculada automaticamente
        print("activacion recibida")
        print(f'reL:{request.labels}')
        labels      = tf.convert_to_tensor(request.labels, dtype=tf.float32)
        
        global epoch
        print("aqui")
        distance = euclidean_distance(activations, activations)
        print(f'dista:{distance}')
        print(f'labels:{labels}')
        loss = contrastive_loss(labels, distance)
        
        print("BACKWARD")
        #server_gradients = tape.gradient(loss, self.server_model.trainable_variables)
        #self.optimizer.apply_gradients(zip(server_gradients, self.server_model.trainable_variables))
        
        #activations_gradients = tape.gradient(loss, activations)
        response              = pb2.ServerToClient()

        #response.gradients.extend(activations_gradients.numpy().flatten())
        response.loss = loss.numpy()
        #response.acc  = acc.numpy()
        epoch        += 1
        
        print(f"Epoch {epoch} - Loss: {loss.numpy()} ")
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
