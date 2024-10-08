# Imports
import random
import warnings

# Filter warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

def euclidean_distance(vects):
    x, y = vects  # Desempaquetar los tensores x e y desde vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)  # Operaciones entre tensores
    return K.sqrt(K.maximum(sum_square, K.epsilon())) 

def load_Client_data():
    # Cargar los datos
    path_data = f'fashion_mnist_pairs.npz'
    data_cliente = np.load(path_data)

    x_train_client = data_cliente['x_train']
    y_train_client = data_cliente['y_train']
    x_test_client = data_cliente['x_test']
    y_test_client = data_cliente['y_test']
    return x_train_client,y_train_client,x_test_client,y_test_client


def show_prediction(pairs, diss, pred):
    """
    Esta función muestra 5 pares de imágenes, cada par con la
    puntuación de disimilitud y la predicción hecha por el modelo.
    """
    # Obtenemos un índice inicial aleatorio
    start_idx = random.randrange(0, len(pairs)-5)  # Aseguramos que haya suficientes pares para mostrar

    # Creamos una figura y subplots para 5 pares (2 filas, 5 columnas)
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))

    # Iteramos sobre los 5 pares
    for i in range(5):
        imagen1 = pairs[start_idx + i][0]
        imagen2 = pairs[start_idx + i][1]
        
        # Mostramos la primera imagen en la fila superior
        axs[0, i].imshow(imagen1)
        axs[0, i].axis('off')  # Desactivamos los ejes

        # Mostramos la segunda imagen en la fila inferior
        axs[1, i].imshow(imagen2)
        axs[1, i].axis('off')  # Desactivamos los ejes
        
        # Añadimos el texto de la disimilitud y la predicción
        axs[0, i].set_title(f"Diss: {diss[start_idx + i]:.2f}\n", fontsize=8)

    # Mostramos la figura con ambas imágenes
    plt.tight_layout()
    plt.show()

# Cargar el modelo
modelo = load_model('siamese_model_client_1.h5')

# Verifica el modelo
#modelo.summary() 

#Crear el modelo global
input_shape = (28, 28, 1)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# 3. Pasar ambas entradas por la subred preentrenada (las dos ramas)
# Aquí la misma subred se reutiliza en ambas ramas de la red siamesa
output_a = modelo(input_a)
output_b = modelo(input_b)

# 5. Usar la capa Lambda para calcular la distancia
distance = Lambda(euclidean_distance, output_shape=(1,))([output_a, output_b])
# 6. Crear el modelo global que tomará las dos imágenes de entrada y generará la distancia
modelo_siames = Model(inputs=[input_a, input_b], outputs=distance)

modelo_siames.summary()
#plot_model(modelo_siames, show_shapes=True, show_layer_names=True)
train_pairs, train_pairs_labels,test_pairs, test_pairs_labels = load_Client_data()

# Prediction
dissimilarity = modelo_siames.predict([test_pairs[:,0], test_pairs[:,1]])

# Compute the accuracy of the model
y_pred = dissimilarity.ravel() < 0.5
acc = np.mean(y_pred == test_pairs_labels)
print(f"The model accuracy is: {acc * 100:.2f}%")

show_prediction(test_pairs, dissimilarity.ravel(), y_pred)
