from imports import tf
from unguided_diffusion.model_blocks import UNetBlocks
from constants import VQVAE_OUTPUT_SHAPE, VQVAE_NUM_BLOCKS_WITH_ACTION

def create_unet(
      input_shape,
      start_filters = 256,
      num_blocks = 2,
      dropout_prob = 0.3,
      filter_size = 3,
      num_channels = 1,
      leaky = 0.05,
      output_channels = None,
      use_action_embedding = False,
      action_embedding_shape = VQVAE_OUTPUT_SHAPE,
      num_blocks_with_action = VQVAE_NUM_BLOCKS_WITH_ACTION):
  """
  Creates U-Net with adjustable parameters.

  Inputs:
  - input_shape | Tuple : shape of input sample.
  - start_filters | int : number of filters for each convolutional layer.
  - num_blocks | int : number of encoder/decoder blocks in U-Net. Each block downsamples/upsamples input.
  - dropout_prob | float : dropout percentage for each encoder/decoder block.
  - filter_size | int : kernel size for each convolutional layer.
  - num_channels | int : number of channels for the input. Also used for output if output_chanenls is None.
  - output_channels | int : number of desired output channels.
  - use_action_embedding | bool : whether this U-Net is conditioned on action embeddings.
  - action_embedding_shape | Tuple : shape of action embedding input.
  - num_blocks_with_action | int : number of decoder blocks that is conditioned with action embeddings.

  Outputs:
  - tf.keras.Model : U-Net model.
  """

  input_block = tf.keras.layers.Input(shape = input_shape)

  if use_action_embedding:
     action_input = tf.keras.layers.Input(shape = action_embedding_shape)
  
  embed_inputs = []
  
  for block in range(num_blocks):
      embed_inputs.append(tf.keras.layers.Input(shape = (1, 1, start_filters), name=f"time_embed_input_{block}"))

  x = input_block

  skip_connections = []

  for block in range(num_blocks):
     x, skip = UNetBlocks.EncoderMiniBlock(
        x, filter_size = filter_size, n_filters = start_filters, dropout_prob = dropout_prob, leaky = leaky)
     skip_connections.append(skip)

  # middle block
  x, _ = UNetBlocks.EncoderMiniBlock(
      x, filter_size = filter_size, n_filters = start_filters, dropout_prob = dropout_prob, max_pooling = False, leaky = leaky, name = "UNET_MIDDLE_BLOCK")

  for block in range(num_blocks):
    s = tf.shape(x)
    if use_action_embedding and (block < num_blocks_with_action):
       x = tf.concat([x, tf.broadcast_to(action_input, tf.shape(x))], axis = -1)
    broadcasted_time = tf.broadcast_to(embed_inputs[block], (s[0], s[1], s[2], start_filters))
    x = tf.concat([x, broadcasted_time], axis = -1) # TODO: try sum
    x = UNetBlocks.DecoderMiniBlock(x, skip_connections[-(block+1)], n_filters = start_filters, filter_size = filter_size, leaky = leaky)
    
  x, _ = UNetBlocks.EncoderMiniBlock(x, filter_size, start_filters, dropout_prob, max_pooling = False, leaky = leaky)

  if output_channels is None:
    output_layer = tf.keras.layers.Conv2D(num_channels, filter_size, padding = "same")(x)
  else:
    output_layer = tf.keras.layers.Conv2D(output_channels, filter_size, padding = "same")(x)

  if use_action_embedding:
     return tf.keras.models.Model(inputs = [input_block, action_input, *embed_inputs], outputs = output_layer)
  return tf.keras.models.Model(inputs = [input_block, *embed_inputs], outputs = output_layer)
