# Partie 3 
import tensorflow as tf
import numpy as np

# Chargement et préparation des données MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Ajout d'une dimension de canal
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Construction du modèle

# couche d'entrée dans le réseau de neurones.
input_layer = tf.keras.layers.Input(shape=(None, None, 1))

# Construction en couche conventionelle
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(input_layer)
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(conv_layer)
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(conv_layer)
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(conv_layer)

conv_output_layer = tf.keras.layers.Conv2D(10, kernel_size=(1, 1))(conv_layer)

# GlobalAveragePooling pour convertir en vecteur de taille 10
global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()(conv_output_layer)


# Création du modèle
model = tf.keras.models.Model(inputs=[input_layer], outputs=[global_avg_layer])

# Compilation du modèle
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Résumé du modèle
model.summary()

# Entraînement du modèle
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=5
)
