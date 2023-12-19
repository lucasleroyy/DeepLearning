import tensorflow as tf
import numpy as np

# Chargement et préparation des données MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Ajout d'une dimension de canal
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Construction du modèle avec couches résiduelles

input_layer = tf.keras.layers.Input(shape=(None, None, 1))

# Fonction pour créer un bloc convolutionnel avec résiduel
def conv_block(input_tensor, num_filters, kernel_size):
    x = input_tensor

    # Sauvegarde de l'entrée pour la connexion résiduelle
    input_to_block = x

    # Bloc convolutionnel
    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, padding='same', activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Ajout de la connexion résiduelle
    x = tf.keras.layers.Add()([x, input_to_block])

    return x

# Application des blocs convolutionnels résiduels
x = conv_block(input_layer, 128, (1, 1))
x = conv_block(x, 128, (7, 7))
x = conv_block(x, 128, (7, 7))
x = conv_block(x, 128, (7, 7))

conv_output_layer = tf.keras.layers.Conv2D(10, kernel_size=(1, 1))(x)

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
