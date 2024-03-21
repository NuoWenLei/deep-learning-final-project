from imports import tf
from model_blocks import UNetBlocks

def create_unet(input_shape, start_filters = 256, num_blocks = 2, dropout_prob = 0.3, filter_size = 3, num_channels = 1, leaky = 0.05, output_channels = None):

  input_block = tf.keras.layers.Input(shape = input_shape)
  
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
      x, filter_size = filter_size, n_filters = start_filters, dropout_prob = dropout_prob, max_pooling = False, leaky = leaky)
  
  for block in range(num_blocks):
    s = tf.shape(x)
    broadcasted_time = tf.broadcast_to(embed_inputs[block], (s[0], s[1], s[2], start_filters))
    x = tf.concat([x, broadcasted_time], axis = -1) # TODO: try sum
    x = UNetBlocks.DecoderMiniBlock(x, skip_connections[-(block+1)], n_filters = start_filters, filter_size = filter_size, leaky = leaky)
    
  x, _ = UNetBlocks.EncoderMiniBlock(x, filter_size, start_filters, dropout_prob, max_pooling = False, leaky = leaky)

  if output_channels is None:
    output_layer = tf.keras.layers.Conv2D(num_channels, filter_size, padding = "same")(x)
  else:
    output_layer = tf.keras.layers.Conv2D(output_channels, filter_size, padding = "same")(x)

  return tf.keras.models.Model(inputs = [input_block, *embed_inputs], outputs = output_layer)
