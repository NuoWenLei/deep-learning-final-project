from imports import tf

class UNetBlocks:

  def EncoderMiniBlock(inputs, filter_size = 3, n_filters=32, dropout_prob=0.3, max_pooling=True, leaky = 0.03, **kwargs):

    layers = [
       tf.keras.layers.Conv2D(n_filters,
                  filter_size,  # filter size
                  activation=tf.keras.layers.LeakyReLU(alpha=leaky),
                  padding='same'),
        tf.keras.layers.Conv2D(n_filters,
                  filter_size,  # filter size
                  activation=tf.keras.layers.LeakyReLU(alpha=leaky),
                  padding='same'),
        tf.keras.layers.BatchNormalization(),
    ]

    if dropout_prob > 0:
       layers.append(tf.keras.layers.Dropout(dropout_prob))
    
    seq = tf.keras.Sequential(layers, **kwargs)
    conv_result = seq(inputs)

    if max_pooling:
       next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv_result)
    else:
       next_layer = conv_result

    skip_connection_layer = conv_result
    return next_layer, skip_connection_layer

  def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32, filter_size = 3, leaky = 0.03):
    up = tf.keras.layers.Conv2DTranspose(
                 n_filters,
                 filter_size,
                 strides=(2,2),
                 padding='same')(prev_layer_input)
    merge = tf.concat([up, skip_layer_input], axis=3)
    conv = tf.keras.layers.Conv2D(n_filters,
                 filter_size,
                 activation=tf.keras.layers.LeakyReLU(alpha=leaky),
                 padding='same')(merge)
    conv = tf.keras.layers.Conv2D(n_filters,
                 filter_size,
                 activation=tf.keras.layers.LeakyReLU(alpha=leaky),
                 padding='same')(conv)
    return conv
  
class PosEmbedding(tf.keras.layers.Layer):
  def __init__(self, dim):
    super(PosEmbedding, self).__init__()
    self.dim = dim

  def call(self, time):
    dim = self.dim/2

    positions = tf.cast(tf.expand_dims(time, -1), tf.float32)
    dims = tf.expand_dims(tf.range(dim), 0)/dim

    angle_rates = 1 / (10000**dims)
    angle_rads = positions * angle_rates

    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


class TimeEmbedding2D(tf.keras.models.Model):
  def __init__(self, hidden_dim, out_dim, **kwargs):
    super(TimeEmbedding2D, self).__init__(**kwargs)
    self.pos_embed = PosEmbedding(hidden_dim)
    self.out_dim = out_dim
    self.dense = tf.keras.layers.Dense(out_dim, name = "TimeEmbedding_dense")

  def call(self, input, **kwargs):
    pos = self.pos_embed(input) # Output shape: (noise_level, hidden_dim)
    dense_output = self.dense(pos) # Output shape: (noise_level, out_dim)
    return dense_output