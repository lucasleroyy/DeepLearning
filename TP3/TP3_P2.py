import tensorflow as tf
import numpy as np

# Chargement du jeu de données MNIST qui contient des images de chiffres manuscrits (0-9).
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisation des images en divisant par 255 pour obtenir des valeurs de pixel entre 0 et 1.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Ajout d'une dimension de canal aux images (nécessaire pour les couches de convolution).
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Construction du modèle de réseau de neurones convolutionnel (CNN).
input_layer = tf.keras.layers.Input(shape=(28, 28, 1))  # Couche d'entrée avec la forme des images MNIST.

# Ajout de plusieurs couches de convolution avec activation ReLU.
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(input_layer)
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(conv_layer)
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(conv_layer)
conv_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu')(conv_layer)

# Aplatir les caractéristiques en un vecteur pour la couche de sortie.
flatten_layer = tf.keras.layers.Flatten()(conv_layer)

# Couche de sortie avec activation softmax pour la classification multiclasse.
output_layer = tf.keras.layers.Dense(10, activation='softmax')(flatten_layer)

# Assemblage du modèle en spécifiant les entrées et sorties.
model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

# Compilation du modèle avec un optimiseur, une fonction de perte et des métriques de performance.
model.compile(
    optimizer="adam",  # Optimiseur Adam pour la descente de gradient.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Fonction de perte pour la classification.
    metrics=["accuracy"]  # Suivi de la précision du modèle.
)

# Affichage du résumé du modèle pour visualiser sa structure.
model.summary()

# Entraînement du modèle sur le jeu de données MNIST.
model.fit(
    x_train, y_train,                 # Données d'entraînement et étiquettes correspondantes.
    validation_data=(x_test, y_test), # Données de validation pour évaluer les performances.
    batch_size=32,                    # Taille de lot pour l'entraînement.
    epochs=5                          # Nombre d'itérations sur l'ensemble des données d'entraînement.
)
