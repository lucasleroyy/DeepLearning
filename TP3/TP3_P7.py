import tensorflow as tf

# Chargement et préparation des données CIFAR-10
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Définir la couche d'entrée pour les images CIFAR-10
input_layer = tf.keras.layers.Input(shape=(32, 32, 3), name="input_layer")

# Nombre réduit de filtres
n = 32

# Fonction pour créer un bloc convolutionnel séquentiellement séparé avec réduction et restauration de canaux
def encoder_decoder_conv_block(input_tensor, num_filters, reduced_filters):
    x = input_tensor

    # Réduction de canaux
    x = tf.keras.layers.Conv2D(filters=reduced_filters, kernel_size=(1, 1), padding="same")(x)

    # Bloc convolutionnel séquentiellement séparé
    for _ in range((9-1)//2):
        x = tf.keras.layers.Conv2D(filters=reduced_filters, kernel_size=(3, 1), padding="same")(x)
        x = tf.keras.layers.Conv2D(filters=reduced_filters, kernel_size=(1, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("tanh")(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    # Restauration des canaux
    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(1, 1), padding="same")(x)
    return x

# Appliquer les blocs convolutionnels avec encodeur/décodeur
x = encoder_decoder_conv_block(input_layer, 128, n)
x = encoder_decoder_conv_block(x, 128, n)
x = encoder_decoder_conv_block(x, 128, n)
x = encoder_decoder_conv_block(x, 128, n)

# Dernière couche Conv2D avant le GlobalAveragePooling
x = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), activation='relu', padding='same')(x)

# GlobalAveragePooling2D
gap_layer = tf.keras.layers.GlobalAveragePooling2D()(x)

# Couche d'activation finale
output_layer = tf.keras.layers.Activation("softmax", name="output_layer")(gap_layer)

# Créer le modèle
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compiler le modèle
model.compile(optimizer="Adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

# Résumé du modèle
model.summary()

# Entraîner le modèle
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=5)
