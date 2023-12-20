import tensorflow as tf
import numpy as np

# Charger l'ensemble de données CIFAR-10
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normaliser les valeurs des pixels entre 0 et 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Définir la couche d'entrée avec la forme des images CIFAR-10
input_layer = tf.keras.layers.Input(shape=(32, 32, 3), name="input_layer")

# Le nombre de filtres et la taille encodée peuvent être ajustés en fonction des besoins du modèle
original_c = 128  # Le nombre original de filtres
encoded_c = 32    # Le nombre réduit de filtres pour l'encodage

x = tf.keras.layers.Conv2D(filters=original_c, kernel_size=(1, 1), padding='same')(input_layer)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)

# Appliquer la même logique pour les autres blocs de convolution
for i in range(2, 5):  # Répéter pour les blocs 2, 3 et 4
    input_to_block = x

    x = tf.keras.layers.Conv2D(filters=encoded_c, kernel_size=(1, 1), padding='same')(x)
    for _ in range((7-1)//2):  # Utilisez (7-1)//2 pour simuler une convolution 7x7
        x = tf.keras.layers.Conv2D(filters=encoded_c, kernel_size=(3, 1), padding="same")(x)
        x = tf.keras.layers.Conv2D(filters=encoded_c, kernel_size=(1, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("tanh")(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(filters=original_c, kernel_size=(1, 1), padding='same')(x)
    x = x + input_to_block  # Ajout de la résiduelle

conv_layer = tf.keras.layers.Conv2D(filters=10, kernel_size=(7,7), activation='relu', padding='same')(x)

# GlobalAveragePooling2D au lieu de Flatten
gap_layer = tf.keras.layers.GlobalAveragePooling2D()(conv_layer)

# Couche d'activation finale
output_layer = tf.keras.layers.Activation("softmax", name="output_layer")(gap_layer)

# Créer le modèle
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compiler le modèle
model.compile(optimizer="Adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

# Afficher un résumé du modèle
model.summary()

# Entraîner le modèle
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=5)
