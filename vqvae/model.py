from imports import tf
from constants import VQVAE_COMMITMENT_COST, VQVAE_DECAY, VQVAE_NUM_EMBEDDINGS, VQVAE_EMBEDDING_DIM, VQVAE_INPUT_SHAPE, LATENT_SHAPE, NUM_PREV_FRAMES
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

	conv1 = tf.keras.layers.Conv2D(latent_dim, 3, padding = "same")(encoder_inputs)
	conv1 = tf.keras.layers.LeakyReLU()(conv1)
	conv1 = tf.keras.layers.BatchNormalization()(conv1)
	# conv1 = tf.keras.layers.MaxPool2D()(conv1)
	
	conv2 = tf.keras.layers.Conv2D(latent_dim, 3, padding = "same")(conv1)
	conv2 = tf.keras.layers.LeakyReLU()(conv2)
	conv2 = tf.keras.layers.BatchNormalization()(conv2)
	# conv2 = tf.keras.layers.MaxPool2D()(conv2)

	x = conv2

	for i in range(num_resblocks):
		
		x = ResBlock(latent_dim, bn = batchnorm, name = f"{name}_resblock{i}")(x)
		if batchnorm:
			x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.MaxPool2D()(x)
	
	x = tf.keras.layers.Conv2D(latent_dim, 3, padding = "same")(x)

	return tf.keras.Model(encoder_inputs, x, name=name)

class ImageVQEncoder(tf.keras.Model):
	"""
	Image Vector Quantizer Encoder.
	"""

	def __init__(self, latent_dim=VQVAE_EMBEDDING_DIM,
		num_embeddings=VQVAE_NUM_EMBEDDINGS,
		image_shape=VQVAE_INPUT_SHAPE[:2],
		num_channels = VQVAE_INPUT_SHAPE[-1],
		ema = True,
		batchnorm = True,
		name = "vq_vae"):

		super().__init__(name = name)

		if ema:
			self.vq_layer = VectorQuantizerEMA(
				embedding_dim = latent_dim, 
				num_embeddings = num_embeddings,
				commitment_cost=VQVAE_COMMITMENT_COST,
				decay=VQVAE_DECAY,
				name="vector_quantizer")
		else:
			self.vq_layer = VectorQuantizer(
				embedding_dim = latent_dim, 
				num_embeddings = num_embeddings,
				commitment_cost=VQVAE_COMMITMENT_COST,
				name="vector_quantizer")
			
		self.encoder = get_encoder(latent_dim = latent_dim, input_shape=image_shape + (num_channels,), batchnorm=batchnorm)

	def get_vq_layer(self):
		return self.vq_layer
	
	def call(self, inputs, step):
		encoder_output = self.encoder(inputs)
		quantized_latents, original_encoding_indices = self.vq_layer(encoder_output, step)
		return quantized_latents, original_encoding_indices
