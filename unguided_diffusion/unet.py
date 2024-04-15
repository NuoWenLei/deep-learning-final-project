from imports import tf
from unguided_diffusion.model_blocks import UNetBlocks
from constants import VQVAE_INPUT_SHAPE, VQVAE_OUTPUT_SHAPE, VQVAE_NUM_BLOCKS_WITH_ACTION

def create_unet(input_shape, start_filters = 256, num_blocks = 2, dropout_prob = 0.3, filter_size = 3, num_channels = 1, leaky = 0.05, output_channels = None, use_action_embedding = False, action_embedding_shape = VQVAE_OUTPUT_SHAPE, num_blocks_with_action = VQVAE_NUM_BLOCKS_WITH_ACTION):

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
       x = x + action_input
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

def create_unet_with_encoder_decoder(input_shape, start_filters = 256, num_blocks = 2, dropout_prob = 0.3, filter_size = 3, num_channels = 1, leaky = 0.05, output_channels = None, use_action_embedding = False, action_embedding_shape = VQVAE_OUTPUT_SHAPE, num_blocks_with_action = VQVAE_NUM_BLOCKS_WITH_ACTION):
   input_block = tf.keras.layers.Input(shape = input_shape)

   if use_action_embedding:
     action_input = tf.keras.layers.Input(shape = action_embedding_shape)

   embed_inputs = []
  
   for block in range(num_blocks):
      embed_inputs.append(tf.keras.layers.Input(shape = (1, 1, start_filters), name=f"time_embed_input_unet_enc_dec_{block}"))
   
   encoder, skip_shapes = get_unet_encoder(input_shape,
                                           start_filters=start_filters,
                                           num_blocks=num_blocks,
                                           dropout_prob=dropout_prob,
                                           filter_size = filter_size,
                                           leaky = leaky)
   
   encoded_state = encoder(input_block)
   decoder = get_unet_decoder(skip_shapes,
                              VQVAE_INPUT_SHAPE,
                              start_filters = start_filters,
                              num_blocks = num_blocks,
                              dropout_prob = dropout_prob,
                              filter_size = filter_size,
                              num_channels=num_channels,
                              leaky = leaky,
                              output_chanenls = output_channels,
                              use_action_embedding= use_action_embedding,
                              action_embedding_shape=action_embedding_shape,
                              num_blocks_with_action=num_blocks_with_action)
   
   output_layer = decoder([*encoded_state, action_input, *embed_inputs])

   if use_action_embedding:
     return tf.keras.models.Model(inputs = [input_block, action_input, *embed_inputs], outputs = output_layer), encoder, decoder
   return tf.keras.models.Model(inputs = [input_block, *embed_inputs], outputs = output_layer), encoder, decoder


def get_unet_encoder(input_shape, start_filters = 256, num_blocks = 2, dropout_prob = 0.3, filter_size = 3, leaky = 0.05):
  input_block = tf.keras.layers.Input(shape = input_shape)

  x = input_block

  skip_connections = []
  skip_connection_shapes = []

  for block in range(num_blocks):
    x, skip = UNetBlocks.EncoderMiniBlock(
        x, filter_size = filter_size, n_filters = start_filters, dropout_prob = dropout_prob, leaky = leaky)
    skip_connections.append(skip)
    skip_connection_shapes.append(tf.shape(skip))

  # middle block
  x, _ = UNetBlocks.EncoderMiniBlock(
      x, filter_size = filter_size, n_filters = start_filters, dropout_prob = dropout_prob, max_pooling = False, leaky = leaky, name = "UNET_MIDDLE_BLOCK")
  
  return tf.keras.models.Model(inputs = [input_block], outputs = [x, *skip_connections], name = "unet_encoder"), skip_connection_shapes

def get_unet_decoder(skip_connection_shapes, input_shape = VQVAE_INPUT_SHAPE, start_filters = 256, num_blocks = 2, dropout_prob = 0.3, filter_size = 3, num_channels = 1, leaky = 0.05, output_channels = None, use_action_embedding = False, action_embedding_shape = VQVAE_OUTPUT_SHAPE, num_blocks_with_action = VQVAE_NUM_BLOCKS_WITH_ACTION):
  input_block = tf.keras.layers.Input(shape = input_shape)

  x = input_block

  skip_connections = []

  for block in range(num_blocks):
     skip_connections.append(tf.keras.layers.Input(shape = skip_connection_shapes[block], name = f"skip_connection_input_{block}"))

  embed_inputs = []
  
  for block in range(num_blocks):
      embed_inputs.append(tf.keras.layers.Input(shape = (1, 1, start_filters), name=f"time_embed_input_{block}"))

  if use_action_embedding:
     action_input = tf.keras.layers.Input(shape = action_embedding_shape)

  for block in range(num_blocks):
    s = tf.shape(x)
    if use_action_embedding and (block < num_blocks_with_action):
       x = x + action_input
    broadcasted_time = tf.broadcast_to(embed_inputs[block], (s[0], s[1], s[2], start_filters))
    x = tf.concat([x, broadcasted_time], axis = -1) # TODO: try sum
    x = UNetBlocks.DecoderMiniBlock(x, skip_connections[-(block+1)], n_filters = start_filters, filter_size = filter_size, leaky = leaky)
    
  x, _ = UNetBlocks.EncoderMiniBlock(x, filter_size, start_filters, dropout_prob, max_pooling = False, leaky = leaky)

  if output_channels is None:
    output_layer = tf.keras.layers.Conv2D(num_channels, filter_size, padding = "same")(x)
  else:
    output_layer = tf.keras.layers.Conv2D(output_channels, filter_size, padding = "same")(x)

  if use_action_embedding:
     return tf.keras.models.Model(inputs = [input_block, *skip_connections, action_input, *embed_inputs], outputs = output_layer)
  return tf.keras.models.Model(inputs = [input_block, *skip_connections, *embed_inputs], outputs = output_layer, name = "unet_decoder")

# class UNetEncoder(tf.keras.models.Model):
   
#    def __init__(self, input_shape, start_filters = 256, num_blocks = 2, dropout_prob = 0.3, filter_size = 3, num_channels = 1, leaky = 0.05, output_channels = None, use_action_embedding = False, action_embedding_shape = VQVAE_OUTPUT_SHAPE, num_blocks_with_action = VQVAE_NUM_BLOCKS_WITH_ACTION, **kwargs):
#       super().__init__(**kwargs)

#       self.input_shape = input_shape
#       self.start_filters = start_filters
#       self.num_blocks = num_blocks
#       self.dropout_prob = dropout_prob
#       self.filter_size = filter_size
#       self.num_channels = num_channels
#       self.leaky = leaky
#       self.output_channels = output_channels
#       self.use_action_embedding = use_action_embedding
#       self.action_embedding_shape = action_embedding_shape
#       self.num_blocks_with_action = num_blocks_with_action