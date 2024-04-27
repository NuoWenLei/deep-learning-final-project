# Sources:
# - DeepMind Sonnet: https://github.com/deepmind/sonnet/blob/v1/sonnet/python/modules/nets/vqvae.py
# - Keras Examples: https://keras.io/examples/generative/vq_vae/#additional-notes
# - PyTorch Implementation by nadavbh12: https://github.com/nadavbh12/VQ-VAE/blob/a360e77d43ec43dd5a989f057cbf8e0843bb9b1f/vq_vae/auto_encoder.py
# - PyTorch Implementation by anon: https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=UoaeVBKvtOYw

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from imports import tf, kmeans_plusplus
from constants import UNCONDITION_PROB, VQVAE_EXPLORE_STEPS, VQVAE_WARMUP_STEPS

class VectorQuantizer(tf.keras.layers.Layer):
	"""Sonnet module representing the VQ-VAE layer.
	Implements the algorithm presented in
	'Neural Discrete Representation Learning' by van den Oord et al.
	https://arxiv.org/abs/1711.00937
	Input any tensor to be quantized. Last dimension will be used as space in
	which to quantize. All other dimensions will be flattened and will be seen
	as different examples to quantize.
	The output tensor will have the same shape as the input.
	For example a tensor with shape [16, 32, 32, 64] will be reshaped into
	[16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
	independently.
	Args:
		embedding_dim: integer representing the dimensionality of the tensors in the
		quantized space. Inputs to the modules must be in this format as well.
		num_embeddings: integer, the number of vectors in the quantized space.
		commitment_cost: scalar which controls the weighting of the loss terms
		(see equation 4 in the paper - this variable is Beta).
	"""

	def __init__(self, embedding_dim, num_embeddings, commitment_cost,
			   name='vq_layer'):
		super(VectorQuantizer, self).__init__(name=name)
		self.embedding_dim = embedding_dim
		self.num_embeddings = num_embeddings
		self.commitment_cost = commitment_cost
		self.trainable = True
		self.is_training = True
		# self.use_kmeans_late_initialization = True

		# Initialize embedding weights
		initializer = tf.keras.initializers.Zeros()
		self.embeddings = tf.Variable(
					initial_value = initializer(shape = (embedding_dim, num_embeddings)),
					trainable = True, name=f"{name}_embeddings")
		self.num_warmup_steps = VQVAE_WARMUP_STEPS
		self.num_explore_steps = VQVAE_EXPLORE_STEPS

	def get_explore_pct(self, step):
		mean_step = tf.reduce_mean(step)
		if mean_step < self.num_warmup_steps:
			return tf.zeros_like(step)
		if mean_step > (self.num_warmup_steps + self.num_explore_steps):
			return 100.0 * tf.ones_like(step)
		return (step - self.num_warmup_steps) / self.num_explore_steps
	
	def initialize_embeddings(self, x):
		centers_, _ = kmeans_plusplus(x.numpy(), self.num_embeddings)
		self.embeddings.assign(tf.transpose(tf.constant(centers_, dtype = tf.float32)))
				
	def call(self, x, step):
		# Calculate the input shape of the inputs and
		# then flatten the inputs keeping `embedding_dim` intact.
		input_shape = tf.shape(x)
		flattened = tf.reshape(x, [-1, self.embedding_dim])

		mean_step = tf.reduce_mean(step)

		if mean_step >= self.num_warmup_steps:
			if mean_step == self.num_warmup_steps:
				self.initialize_embeddings(flattened)

			explore_pct = self.get_explore_pct(step)

			explore_mask = tf.random.uniform((input_shape[0], ))

			# Quantization.
			original_encoding_indices = self.get_code_indices(flattened)
			random_indices = tf.random.uniform(tf.shape(original_encoding_indices), minval = 1, maxval = self.num_embeddings, dtype = tf.int64)

			# Action exploration decays linearly per steps
			original_encoding_indices = tf.where(
				explore_mask > explore_pct,
				random_indices,
				original_encoding_indices
			)
			if self.is_training:
				# Percentage of unconditional training
				uncondition_mask = tf.random.uniform((input_shape[0], ))
				# random_indices = tf.random.uniform(tf.shape(encoding_indices), minval = 0, maxval = self.num_embeddings, dtype = tf.int64)
				encoding_indices = tf.where(
					uncondition_mask > (UNCONDITION_PROB),
					original_encoding_indices,
					0)
			else:
				encoding_indices = original_encoding_indices
				# encoding_indices = tf.where(uncondition_mask > UNCONDITION_PROB, encoding_indices, 0)
			encodings = tf.one_hot(encoding_indices, self.num_embeddings)
			quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

			# Reshape the quantized values back to the original input shape
			quantized = tf.reshape(quantized, input_shape)

			# Calculate vector quantization loss and add that to the layer. You can learn more
			# about adding losses to different layers here:
			# https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
			# the original paper to get a handle on the formulation of the loss function.
			commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
			codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
			self.add_loss(self.commitment_cost * commitment_loss + codebook_loss)

			# Straight-through estimator.
			quantized = x + tf.stop_gradient(quantized - x)
		else:
			quantized = x
			original_encoding_indices = tf.zeros((input_shape[0], ), dtype = tf.int32)

		return quantized, original_encoding_indices

	def get_code_indices(self, flattened_inputs):
		# Calculate L2-normalized distance between the inputs and the codes.
		similarity = tf.matmul(flattened_inputs, self.embeddings)
		distances = (
			tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
			+ tf.reduce_sum(self.embeddings ** 2, axis=0)
			- 2 * similarity
		)
		max_dist = tf.stop_gradient(tf.reduce_max(distances))
		distances = tf.where(tf.range(self.num_embeddings) > 0, distances, max_dist)
		print(tf.shape(distances))
		# Derive the indices for minimum distances.
		encoding_indices = tf.argmin(distances, axis=1)
		
		return encoding_indices
	
	def quantize(self, encoding_indices):
		return tf.nn.embedding_lookup(self.embeddings, encoding_indices, validate_indices=False)
	
	def reset_embeddings(self, embed_indices):
		col_indices = tf.stop_gradient(tf.stack(tf.meshgrid(tf.range(self.embedding_dim), embed_indices, indexing='ij'), axis=-1))
		tf.stop_gradient(self.embeddings.scatter_nd_update(
			col_indices,
			tf.transpose(self.initializer((tf.shape(embed_indices)[0], self.embedding_dim)))
		))


class VectorQuantizerEMA(tf.keras.layers.Layer):
	"""Sonnet module representing the VQ-VAE layer.
	Implements a slightly modified version of the algorithm presented in
	'Neural Discrete Representation Learning' by van den Oord et al.
	https://arxiv.org/abs/1711.00937
	The difference between VectorQuantizerEMA and VectorQuantizer is that
	this module uses exponential moving averages to update the embedding vectors
	instead of an auxiliary loss. This has the advantage that the embedding
	updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
	...) used for the encoder, decoder and other parts of the architecture. For
	most experiments the EMA version trains faster than the non-EMA version.
	Input any tensor to be quantized. Last dimension will be used as space in
	which to quantize. All other dimensions will be flattened and will be seen
	as different examples to quantize.
	The output tensor will have the same shape as the input.
	For example a tensor with shape [16, 32, 32, 64] will be reshaped into
	[16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
	independently.
	Args:
	embedding_dim: integer representing the dimensionality of the tensors in the
		quantized space. Inputs to the modules must be in this format as well.
	num_embeddings: integer, the number of vectors in the quantized space.
	commitment_cost: scalar which controls the weighting of the loss terms (see
		equation 4 in the paper).
	decay: float, decay for the moving averages.
	epsilon: small float constant to avoid numerical instability.
	"""

	def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay,
			   epsilon=1e-5, name='VectorQuantizerEMA'):
		super(VectorQuantizerEMA, self).__init__(name=name)
		self.embedding_dim = embedding_dim
		self.num_embeddings = num_embeddings
		self.decay = decay
		self.commitment_cost = commitment_cost
		self.epsilon = epsilon
		self.trainable = True

		# Initialize embedding weights
		initializer = tf.keras.initializers.RandomNormal()
		constant_initializer = tf.keras.initializers.Constant(0.)
		embed_init = initializer(shape = (embedding_dim, num_embeddings))
		self.embeddings = tf.Variable(
					initial_value = embed_init,
					name=f"{name}_embeddings")
		self._ema_cluster_size = tf.Variable(
			initial_value = constant_initializer(shape = (num_embeddings)),
			name = 'ema_cluster_size')
		self._ema_w = tf.Variable(
			initial_value = embed_init)

	def call(self, inputs):
		input_shape = tf.shape(inputs)
		flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

		encoding_indices = self.get_code_indices(flat_inputs)
		# quantized = tf.nn.embedding_lookup(self.embeddings, encoding_indices)
		encodings = tf.one_hot(encoding_indices, self.num_embeddings)
		quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
		quantized = tf.reshape(quantized, input_shape)
		e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)

		if self.trainable:
			updated_ema_cluster_size = self._ema_cluster_size * self.decay\
				  + (1 - self.decay) * tf.reduce_sum(encodings, 0)
			dw = tf.matmul(flat_inputs, encodings, transpose_a=True)
			updated_ema_w = self._ema_w.assign(
				self._ema_w * self.decay\
				  + (1 - self.decay) * dw)

			n = tf.reduce_sum(updated_ema_cluster_size)
			updated_ema_cluster_size = self._ema_cluster_size.assign(
				(updated_ema_cluster_size + self.epsilon)
				/ (n + self.num_embeddings * self.epsilon) * n)

			normalised_updated_ema_w = (
				updated_ema_w / tf.reshape(updated_ema_cluster_size, [1, -1]))
			with tf.control_dependencies([e_latent_loss]):
				update_w = self.embeddings.assign(normalised_updated_ema_w)
				with tf.control_dependencies([update_w]):
					loss = self.commitment_cost * e_latent_loss
		else:
			loss = self.commitment_cost * e_latent_loss
		
		self.add_loss(loss)
		quantized = inputs + tf.stop_gradient(quantized - inputs)
		return quantized
	
	def get_code_indices(self, flattened_inputs):
		# Calculate L2-normalized distance between the inputs and the codes.
		similarity = tf.matmul(flattened_inputs, self.embeddings)
		distances = (
			tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
			+ tf.reduce_sum(self.embeddings ** 2, axis=0)
			- 2 * similarity
		)

		# Derive the indices for minimum distances.
		encoding_indices = tf.argmin(distances, axis=1)
		return encoding_indices