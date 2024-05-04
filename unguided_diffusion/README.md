# Unguided Diffusion

## `diffusion.py`

**`class UnguidedVideoDiffusion(tf.keras.models.Model)`**

Diffusion model that does unguided video diffusion using frame-stacking techniques.

**`class LatentActionVideoDiffusion(UnguidedVideoDiffusion)`**

Diffusion model that learns conditioned actions from unguided video diffusion using a Latent Action model.

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
