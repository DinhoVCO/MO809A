import grpc
import siamese_pb2
import siamese_pb2_grpc
from concurrent import futures
import tensorflow as tf
import numpy as np

# Definir la distancia euclidiana
def euclidean_distance(embedding_a, embedding_b):
    return np.sqrt(np.sum((embedding_a - embedding_b) ** 2))

# Definir la pérdida contrastiva
def contrastive_loss(y_true, distance, margin=1.0):
    positive_loss = y_true * tf.square(distance)
    negative_loss = (1 - y_true) * tf.square(tf.maximum(margin - distance, 0))
    return tf.reduce_mean(positive_loss + negative_loss)

# Definir el servidor gRPC
class SiameseServicer(siamese_pb2_grpc.SiameseServicer):
    def __init__(self):
        self.embeddings = {}
        self.labels = {}  # Aquí se pueden almacenar etiquetas (si es necesario)
        self.siamese_branch = self.create_siamese_branch((28, 28, 1))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def create_siamese_branch(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu")
        ])
        return model

    def SendEmbedding(self, request, context):
        client_id = request.client_id
        embedding = np.array(request.embedding)

        # Guardar los embeddings recibidos
        self.embeddings[client_id] = embedding

        # Cuando ambos embeddings estén disponibles, calcular la distancia
        if len(self.embeddings) == 2:
            embedding_a = self.embeddings['client_1']
            embedding_b = self.embeddings['client_2']

            # Simulación de una etiqueta (0 = similar, 1 = disimilar)
            y_true = 0  # Se asume que el par es similar

            # Calcular la distancia euclidiana entre los embeddings
            distance = euclidean_distance(embedding_a, embedding_b)
            print(f"Distancia entre embeddings: {distance}")

            # Calcular la pérdida
            #with tf.GradientTape() as tape:
            loss = contrastive_loss(y_true, distance)
            print(f"Pérdida: {loss.numpy()}")

            # Calcular los gradientes y actualizar los pesos
            #grads = tape.gradient(loss, self.siamese_branch.trainable_weights)
            #self.optimizer.apply_gradients(zip(grads, self.siamese_branch.trainable_weights))
            #print("aqui")
            # Extraer los nuevos pesos después de la actualización
            #updated_weights = [w.numpy().flatten().tolist() for w in self.siamese_branch.trainable_weights]

            # Responder con los nuevos pesos
            return siamese_pb2.EmbeddingResponse(
                message=f"Enviando loss. {client_id}",
                loss=loss.numpy()
            )

        return siamese_pb2.EmbeddingResponse(
            message=f"Embedding recibido de {client_id}, esperando otro cliente"
        )

# Iniciar el servidor
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    siamese_pb2_grpc.add_SiameseServicer_to_server(SiameseServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Servidor siamesa escuchando en el puerto 50051.")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
