# Unguided Diffusion

See [this Google Colab noteboook](https://colab.research.google.com/drive/1gWPoUnxKtZOkXORMHY5T0bJ3QIcpZk68?usp=sharing) for a demo.

## `diffusion.py`

**`class UnguidedDiffusion(tf.keras.models.Model)`**

Diffusion model class that does not take any guidance or context and is optimized with the Score Matching loss function.

The class also contains method for unguided sampling with Annealed Langevin Dynamics.

## `model_blocks.py`

**`class UNetBlocks`**

A class that contains methods to create Encoder and Decoder UNet blocks.

**`class PosEmbedding`**

Custom Keras layer that embeds positional information to the input via positional embedding methods with sine and cosine.

**`class TimeEmbedding2D`**

Custom Keras model that embeds positional information to the input with `PosEmbedding` and further projects the embedded input into a new dimension.

## `unet.py`

**`def create_unet(input_shape, start_filters = 256, num_blocks = 2, dropout_prob = 0.3, filter_size = 3, num_channels = 1, leaky = 0.05)`**

Function that constructs a UNet Keras model conditioned on some context embeddings via concatenation.

## `helpers.py`

**`def create_flow_unguided(x, batch_size = 128, preprocess_func = None, repeat = True)`**

Function that instantiates a data generator that yields batches of a certain batch size preprocessed per batch.
