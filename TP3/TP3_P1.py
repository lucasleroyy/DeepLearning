import tensorflow as tf
import numpy as np

# Charger le dataset MNIST, qui est une collection standard d'images manuscrites de chiffres.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les données en convertissant les valeurs de pixels de 0-255 à 0-1 pour améliorer l'apprentissage.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Créer la couche d'entrée du modèle. Chaque image d'entrée a une forme de 28x28 pixels.
input_layer = tf.keras.layers.Input(name="input_layer", shape=(28, 28))

# Aplatir les images 2D en vecteurs 1D pour les traiter dans des couches denses.
flatten_layer = tf.keras.layers.Flatten()(input_layer)

# Ajouter une première couche dense (fully connected) avec 128 neurones.
hidden_layer = tf.keras.layers.Dense(units=128)(flatten_layer)

# Utiliser la fonction d'activation ReLU pour introduire de la non-linéarité.
hidden_layer = tf.keras.layers.Activation("relu")(hidden_layer)

# Ajouter une couche de Dropout pour réduire le surajustement en "éteignant" aléatoirement 20% des neurones pendant l'entraînement.
hidden_layer = tf.keras.layers.Dropout(0.2)(hidden_layer)

# Ajouter une deuxième couche dense pour la classification finale. Elle a 10 unités, une pour chaque classe de chiffre (0-9).
hidden_layer = tf.keras.layers.Dense(units=10)(hidden_layer)

# Utiliser la fonction d'activation softmax pour la couche de sortie, ce qui est courant dans les problèmes de classification multiclasse.
output_layer = tf.keras.layers.Activation("softmax", name="output_layer")(hidden_layer)

# Créer le modèle en spécifiant les couches d'entrée et de sortie.
model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

# Afficher un résumé du modèle pour avoir un aperçu de l'architecture et du nombre de paramètres.
model.summary(150)

# Compiler le modèle en spécifiant l'optimiseur, la fonction de perte et les métriques pour l'évaluation.
model.compile(
    optimizer="Adam",
    loss={"output_layer": tf.keras.losses.SparseCategoricalCrossentropy()},
    metrics=["acc"],
)

# Entraîner le modèle sur les données d'entraînement, tout en validant sur les données de test.
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=5
)
