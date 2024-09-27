import grpc
import siamese_pb2
import siamese_pb2_grpc
import tensorflow as tf
import numpy as np

# Definir la rama del modelo siamesa
def create_siamese_branch(input_shape):
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
def get_activations(siamese_branch, X):
    with tf.GradientTape(persistent=True) as tape:
        embedding = siamese_branch(np.expand_dims(X, axis=0)).numpy().flatten()
    return embedding, tape
# Configurar el cliente
def run_client(image, client_id, server_address='localhost:50051'):
    # Crear la rama siamesa
    input_shape = (28, 28, 1)  # Ejemplo para Fashion MNIST
    siamese_branch = create_siamese_branch(input_shape)
    embedding, tape=get_activations(siamese_branch,image)

    # Conectarse al servidor
    channel = grpc.insecure_channel(server_address)
    stub = siamese_pb2_grpc.SiameseStub(channel)

    # Enviar el embedding al servidor
    embedding_request = siamese_pb2.EmbeddingRequest(embedding=embedding, client_id=client_id)
    response = stub.SendEmbedding(embedding_request)
    print(f"Cliente {client_id}: {response.message}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # Recibir los pesos actualizados (si se proporcionan)
    if response.loss:
        print(f"Cliente {client_id}: loss recibidos")
        # Calcular los gradientes y actualizar los pesos
        loss_tensor = tf.convert_to_tensor(response.loss, dtype=tf.float32)
        grads = tape.gradient(loss_tensor, siamese_branch.trainable_weights)
        optimizer.apply_gradients(zip(grads, siamese_branch.trainable_weights))

if __name__ == "__main__":
    # Aquí se podría cargar una imagen de ejemplo para este cliente
    dummy_image = np.random.rand(28, 28, 1)  # Imagen ficticia
    run_client(dummy_image, client_id="client_1")
