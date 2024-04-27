from imports import tf
from constants import VQVAE_COMMITMENT_COST, VQVAE_DECAY, VQVAE_NUM_EMBEDDINGS, VQVAE_EMBEDDING_DIM, VQVAE_INPUT_SHAPE
from vqvae.quantizers import VectorQuantizer, VectorQuantizerEMA

class ResBlock(tf.keras.layers.Layer):
	"""
	Convolutional Residual Block for Convolutional Variational Auto-Encoder
	"""
	def __init__(self, out_channels, mid_channels=None, bn=False, name = None):
		super(ResBlock, self).__init__(name = name)

		if mid_channels is None:
			mid_channels = out_channels

		layers = [
			tf.keras.layers.Activation(tf.keras.activations.relu),
			tf.keras.layers.Conv2D(mid_channels, kernel_size = 3, strides = 1, padding = "same", use_bias = True),
			tf.keras.layers.Activation(tf.keras.activations.relu),
			tf.keras.layers.Conv2D(out_channels, kernel_size = 1, strides = 1, padding = "valid", use_bias = True)
		]

		if bn:
			layers.insert(2, tf.keras.layers.BatchNormalization())

		self.convs = tf.keras.Sequential(layers)

	def call(self, x):
		# Residual output
		return x + self.convs(x)


def get_encoder(latent_dim=VQVAE_EMBEDDING_DIM, input_shape=VQVAE_INPUT_SHAPE, num_resblocks = 2, batchnorm = True, name="encoder"):
	"""
	Construct Convolutional Encoder
	Args:
		- latent_dim = EMBEDDING_DIM: embedding size for auto-encoder
		- input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3): shape of input
		- num_resblocks = 2: number of residual convolution blocks
		- batchnorm = True: use of BatchNormalization layers in residual blocks
		- name = "encoder": name of model
	Returns:
		- tensorflow.keras.Model
	"""
	encoder_inputs = tf.keras.Input(shape=input_shape)

	conv1 = tf.keras.layers.Conv2D(latent_dim, 4, strides = 2, padding = "same", use_bias = False)(encoder_inputs)
	conv1 = tf.keras.layers.LeakyReLU()(conv1)
	conv1 = tf.keras.layers.BatchNormalization()(conv1)
	
	conv2 = tf.keras.layers.Conv2D(latent_dim, 4, strides = 2, padding = "same", use_bias = False)(conv1)
	conv2 = tf.keras.layers.LeakyReLU()(conv2)
	conv2 = tf.keras.layers.BatchNormalization()(conv2)
	x = conv2

	for i in range(num_resblocks):
		x = ResBlock(latent_dim, bn = batchnorm, name = f"{name}_resblock{i}")(x)
		if batchnorm:
			x = tf.keras.layers.BatchNormalization()(x)

	return tf.keras.Model(encoder_inputs, x, name=name)


def get_decoder(input_shape, latent_dim=VQVAE_EMBEDDING_DIM, num_resblocks = 2, num_channels = 3, name="decoder"):
	"""
	Constructs Convolutional Decoder
	Args:
		- input_shape: input shape of decoder
		- latent_dim = EMBEDDING_DIM: embedding size for auto-encoder
		- num_resblocks = 2: number of residual convolution blocks
		- num_channels = 3: number of output channels (RGB)
		- name = "decoder": name of model
	Returns:
		- tensorflow.keras.Model
	"""
	decoder_inputs = tf.keras.Input(shape=input_shape)

	x = tf.keras.layers.Conv2D(latent_dim, kernel_size = 4, strides = 1, padding = "same", use_bias = False)(decoder_inputs)

	for i in range(num_resblocks):
		x = ResBlock(latent_dim, name = f"{name}_resblock{i}")(x)

	conv1 = tf.keras.layers.Conv2DTranspose(latent_dim, kernel_size = 4, strides = 2, padding = "same", use_bias = False)(x)
	conv1 = tf.keras.layers.LeakyReLU()(conv1)
	conv1 = tf.keras.layers.BatchNormalization()(conv1)

	conv2 = tf.keras.layers.Conv2DTranspose(latent_dim, kernel_size=4, strides = 2, padding = "same", use_bias = False)(conv1)
	conv2 = tf.keras.layers.LeakyReLU()(conv2)
	conv2 = tf.keras.layers.BatchNormalization()(conv2)

	decoder_outputs = tf.keras.layers.Conv2DTranspose(num_channels, kernel_size=4, padding = "same", use_bias = False)(conv2)

	decoder_outputs = tf.keras.activations.tanh(decoder_outputs)

	return tf.keras.Model(decoder_inputs, decoder_outputs, name=name)

def get_image_vq_encoder(
		latent_dim=VQVAE_EMBEDDING_DIM,
		num_embeddings=VQVAE_NUM_EMBEDDINGS,
		image_shape=VQVAE_INPUT_SHAPE[:2],
		num_channels = VQVAE_INPUT_SHAPE[-1],
		ema = True,
		batchnorm = True,
		name = "vq_vae"):
	"""
	Create an Image Vector-Quantized Encoder
	"""
	if ema:
		vq_layer = VectorQuantizerEMA(
			embedding_dim = latent_dim, 
			num_embeddings = num_embeddings,
			commitment_cost=VQVAE_COMMITMENT_COST,
			decay=VQVAE_DECAY,
			name="vector_quantizer")
	else:
		vq_layer = VectorQuantizer(
			embedding_dim = latent_dim, 
			num_embeddings = num_embeddings,
			commitment_cost=VQVAE_COMMITMENT_COST,
			name="vector_quantizer")
	# encoder = get_encoder(latent_dim = latent_dim, input_shape=image_shape + (num_channels,), batchnorm=batchnorm)
	
	inputs = tf.keras.Input(shape=image_shape + (num_channels,))
	step = tf.keras.Input(shape=())
	reshaped_inputs = tf.keras.layers.Reshape((1, -1), inputs)

	self_attn = tf.keras.layers.MultiHeadAttention(num_heads = image_shape[0], key_dim = image_shape[1] * num_channels, output_shape=(num_channels, ))
	attn_outputs = self_attn([reshaped_inputs, reshaped_inputs])
	attn_outputs = tf.keras.layers.Reshape((1, 1, num_channels), attn_outputs)

	quantized_latents, original_encoding_indices = vq_layer(attn_outputs, step)

	vq_encoder = tf.keras.Model(inputs = [inputs, step], outputs = [quantized_latents, original_encoding_indices], name=name)
	vq_encoder.build(image_shape + (num_channels,))
	print(vq_encoder.summary())
	return vq_encoder, vq_layer
