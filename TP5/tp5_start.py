import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

from data_generator import SegmentationDataGenerator
from tp5_utils import ResidualConv2D, UNetEncoder, UNetDecoder

BATCHSIZE = 4
DEPTH = 4
PROJ_SIZE = 8
NCLASSES = 1

# Fonction pour les blocs d'encodeur
def encoder_block(input_tensor, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    pooled = tf.keras.layers.MaxPooling2D((2, 2), strides=2)(x)
    return x, pooled

# Fonction pour les blocs de décodeur
def decoder_block(input_tensor, skip_tensor, num_filters):
    x = tf.keras.layers.UpSampling2D((2, 2))(input_tensor)
    x = tf.keras.layers.concatenate([x, skip_tensor])
    x = ResidualConv2D(num_filters, (3, 3))(x)  
    return x

"""Model Code"""
# input
inputs = tf.keras.layers.Input(shape=(None, None, 3), name="images")

# transform
hidden_layer = tf.keras.layers.Conv2D(filters=PROJ_SIZE, kernel_size=(1, 1))(inputs)
hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
hidden_layer = tf.keras.layers.Activation("relu")(hidden_layer)
hidden_layer = tf.keras.layers.Dropout(0.2)(hidden_layer)

# Encoder blocks with UNetEncoder
skip_connections = []
x = hidden_layer
for num_filters in [32, 64, 128, 256][:DEPTH]:  # Utilisation de DEPTH pour contrôler la profondeur
    residual_output, x = UNetEncoder(num_filters)(x)
    skip_connections.append(residual_output)

# Decoder blocks
for i, num_filters in enumerate([256, 128, 64, 32][:DEPTH]): 
    x = decoder_block(x, skip_connections[-(i+1)], num_filters)

# transform back and predict
output_layer = tf.keras.layers.Conv2D(filters=NCLASSES, kernel_size=(1, 1))(x)
output_layer = tf.keras.layers.BatchNormalization()(output_layer)
output_layer = tf.keras.layers.Activation("sigmoid", name="output")(output_layer)

model = tf.keras.models.Model(inputs, output_layer)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics="acc",
)

IMAGE_PATH = glob("dataset/images/*.jpg")
ANNOT_PATH = glob("dataset/labels/*.jpg")

data_generator_train = SegmentationDataGenerator(
    IMAGE_PATH[:40],
    ANNOT_PATH[:40],
    BATCHSIZE,
    DEPTH,
)
data_generator_test = SegmentationDataGenerator(
    IMAGE_PATH[40:],
    ANNOT_PATH[40:],
    1,
    DEPTH,
)

# Ajout  dun callback pour afficher les métriques d'entraînement
metrics_callback = tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1}: loss={logs['loss']}, acc={logs['acc']}, val_loss={logs['val_loss']}, val_acc={logs['val_acc']}"),
    on_epoch_begin=lambda epoch, logs: print(f"Starting epoch {epoch+1}"),
)

model.fit(
    data_generator_train,
    validation_data=data_generator_test,
    epochs=1000,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            "val_acc",
            patience=50,
            restore_best_weights=True,
            verbose=1,
            start_from_epoch=100,
        ),
        metrics_callback,  
    ],
)

model.evaluate(data_generator_test)

model.save("model")
