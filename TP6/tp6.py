import tensorflow as tf

from tp6_utils import Embeddings, TransformerBlock, TextGen
from tp6_data import load_data


def build_GPT(vocab_size, embed_size, num_heads, num_trblocks, ff_proj):
    inputs = tf.keras.layers.Input((None,), name="input_indexes")

    embeddings_layer = Embeddings(vocab_size, embed_size)(inputs)
    
    transformer_blocks = []
    for _ in range(num_trblocks):
        transformer_block = TransformerBlock(num_heads, ff_proj)(embeddings_layer)
        transformer_blocks.append(transformer_block)
    
    hidden_layers = tf.keras.layers.Concatenate()(transformer_blocks)
    
    outputs = tf.keras.layers.Dense(units=vocab_size, name="out_prediction")(hidden_layers)
    return tf.keras.models.Model([inputs], [outputs])



VOCAB_SIZE = 20000  # Only consider the top vocab_size words
EMBED_SIZE = 512  # Embedding size for each token
NUM_HEADS = 8 # Number of attention heads
NUM_TRAN_BLOCKS = 12 # Number of transformer blocks
LAST_FF_PROJECTION_DIM = (
    2 * EMBED_SIZE
)  # projection size for the last part of transformers
BATCH_SIZE = 64

EPOCHS = 10

text_ds = load_data(VOCAB_SIZE, BATCH_SIZE)

model = build_GPT(
    VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_TRAN_BLOCKS, LAST_FF_PROJECTION_DIM
)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.AdamW(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

model.fit(text_ds, epochs=EPOCHS, callbacks=[TextGen()])

model.save("model")
