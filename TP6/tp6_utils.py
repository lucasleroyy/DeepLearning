import tensorflow as tf
import numpy as np
from tp6_data import OOV_CHAR, WORD_INDEX, INDEX_FROM, INVERTED_WORD_INDEX


class SPE(tf.keras.layers.Layer):
    def __init__(self, output_dims: int = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_dims = output_dims

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dims": self.output_dims,
            }
        )
        return config

    def call(self, inputs):
        # Get the dynamic sequence length from the inputs
        seq_length = tf.shape(inputs)[1]

        # Compute positional encodings
        position = tf.range(seq_length, dtype=tf.float32)
        freqs = tf.range(self.output_dims // 2, dtype=tf.float32)
        freqs = 1 / (10000 ** (2 * freqs / self.output_dims))

        angles = position[:, tf.newaxis] * freqs[tf.newaxis, :]
        pos_encoding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)

        # Extend pos_encoding to match batch size
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.tile(pos_encoding, [tf.shape(inputs)[0], 1, 1])


class Embeddings(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        embed_size,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

    def build(self, input_shape):
        super().build(input_shape)
        self.token_emb = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.embed_size
        )
        self.pos_emb = SPE(output_dims=self.embed_size)

    def call(self, inputs):
        token_embedding = self.token_emb(inputs)  # Token embedding
        pos_embedding = self.pos_emb(inputs)  # Positional embedding
        return token_embedding + pos_embedding



class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, ff_proj, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.ff_proj = ff_proj

    def build(self, input_shape):
        super().build(input_shape)
        embed_dim = input_shape[-1]

        self.multihead_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=embed_dim // self.num_heads
        )
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = tf.keras.layers.Dense(self.ff_proj, activation="gelu")
        self.dense2 = tf.keras.layers.Dense(embed_dim)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, inputs):
        attention_path = self.multihead_attention(inputs, inputs, inputs)
        attention_path = self.layer_norm1(inputs + attention_path)
        dense_path = self.dense1(attention_path)
        dense_path = self.dense2(dense_path)
        dense_path = self.dropout(dense_path)
        output = self.layer_norm2(attention_path + dense_path)
        return output



class TextGen(tf.keras.callbacks.Callback):
    def __prompt(self, text, originality):
        print()
        predict_and_write(self.model, prompt=text, size=50, originality=originality)

    def on_epoch_end(self, epoch, logs=None):
        self.__prompt("This movie is", 1)
        self.__prompt("This movie is", 2)
        self.__prompt("This movie is", 5)
        self.__prompt("This movie is", 10)
        self.__prompt("This movie is", 20)
        print()
        self.model.save(f"model_{epoch}")
        return super().on_epoch_end(epoch, logs)


def predict_and_write(model, prompt: str, size: int, originality: int):
    print(prompt, end=" ")
    prompt = [1] + [  # <- start with the starting symbol [START]
        WORD_INDEX.get(p.lower().strip(), OOV_CHAR - INDEX_FROM)
        + INDEX_FROM  # <- change all words into their indexes
        for p in prompt.split(" ")
        if p != " "
    ]
    for _ in range(size - len(prompt) + 1):
        pred = model(np.asarray(prompt)[np.newaxis, ...])
        pred = tf.keras.activations.softmax(pred)[0, -1, ...]
        topk = tf.math.top_k(pred, k=originality)
        i = np.random.choice(topk.indices.numpy(), 1)[0]
        prompt.append(i)
        print(INVERTED_WORD_INDEX[i], end=" ", flush=True)
