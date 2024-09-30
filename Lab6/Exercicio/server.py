import siamese_pb2 as pb2
import siamese_pb2_grpc as pb2_grpc
import grpc
from concurrent import futures
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
import time
import pandas as pd
import os
from tensorflow.keras import backend as K


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

loss_to_save = []
batch = 0

def euclidean_distance(vects):
    """
    This function calculate the euclidean distance
    between two vectore and return it.
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)

    return K.sqrt(sum_square)

def contrastive_loss(y_true, y_pred, margin=1.0):
        """
        This funtion allow to calculate the loss of the model. Having
        a margin indicates that dissimilar pairs that are beyond this
        margin will not contribute to the loss.
        """
        margin=1.0
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))

        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def createModel(input_shape):
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    # Aplicamos la Lambda layer para calcular la distancia
    distance = Lambda(euclidean_distance)([input_a, input_b])

    # Definimos el modelo final
    model = Model(inputs=[input_a, input_b], outputs=distance)
    return model
class SiameseService(pb2_grpc.SiameseServicer):

    def __init__(self):
        self.server_model = createModel((128,))
        
    def SendClientActivations(self, request, context):

        activations1 = tf.convert_to_tensor(request.activations1, dtype=tf.float32)
        activations1 = tf.reshape(activations1, (request.batch_size, -1)) # -1 faz com que a dimensao seja calculada automaticamente
        activations2 = tf.convert_to_tensor(request.activations2, dtype=tf.float32)
        activations2 = tf.reshape(activations2, (request.batch_size, -1)) 
        labels      = tf.convert_to_tensor(request.labels, dtype=tf.float32)
        global batch

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([activations1,activations2])
            predictions = self.server_model([activations1, activations2])

            loss        = contrastive_loss(labels, predictions)        
        #print("BACKWARD")
        activations_gradients1 = tape.gradient(loss, activations1)
        activations_gradients2 = tape.gradient(loss, activations2)
        response              = pb2.ServerToClient()

        response.gradients1.extend(activations_gradients1.numpy().flatten())
        response.gradients2.extend(activations_gradients2.numpy().flatten())
        response.loss = loss.numpy()
        batch        += 1
        
        print(f"Batch {batch} - Loss: {loss.numpy()} ")
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
    pb2_grpc.add_SiameseServicer_to_server(SiameseService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()
    

if __name__ == '__main__':
    serve()
