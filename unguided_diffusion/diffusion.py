from constants import VQVAE_EMBEDDING_DIM, VQVAE_NUM_EMBEDDINGS, VQVAE_INPUT_SHAPE, VQVAE_LOSS_LAMBDA, CONDITIONAL_SAMPLING_LAMBDA, GRAM_MATRIX_LAMBDA, VQVAE_EXPLORE_STEPS
from imports import tf, np, tqdm
from unguided_diffusion.unet import create_unet
from unguided_diffusion.model_blocks import TimeEmbedding2D
from vqvae.model import get_image_vq_encoder

class UnguidedDiffusion(tf.keras.models.Model):

  def __init__(self, input_shape = (64, 64),
               num_channels = 3,
               batch_size = 32,
               start_filters = 256,
               num_blocks = 3,
               dropout_prob = 0,
               filter_size = 3,
               noise_level = 10,
               leaky = 0.05,
               **kwargs):

    super().__init__(**kwargs)

    self.unet = create_unet(input_shape + (num_channels, ), start_filters, num_blocks, dropout_prob, filter_size, leaky = leaky)
    self.unet.build([input_shape + (num_channels, ), (1, 1, start_filters), (1, 1, start_filters)])
    self.noise_level = noise_level
    self.image_input_shape = input_shape
    self.num_channels = num_channels
    self.full_input_shape = (-1, ) + self.image_input_shape + (self.num_channels, )
    self.total_image_dims = self.image_input_shape[0] * self.image_input_shape[1] * self.num_channels
    self.batch_size = batch_size

    self.time_embeddings = []

    for block in range(num_blocks):
      time_embed = TimeEmbedding2D(start_filters, start_filters, name=f"Time_embedding_block_{block}")
      time_embed.build((1,))
      self.time_embeddings.append(
          time_embed
      )

    self.variance_schedule = self.linear_variance_schedule(noise_level)

  def linear_variance_schedule(self, timesteps):
    lin_start = 1.
    lin_end = 0.001
    return tf.cast(tf.linspace(lin_start, lin_end, timesteps), tf.float32)

  def call(self, x, time_index):

      # Time embedding
      time_context = [time_embed(time_index)[:, tf.newaxis, tf.newaxis, ...] for time_embed in self.time_embeddings]

      return self.unet([x, *time_context])

  def train_step(self, x):

    # Sample noise
    epsilon = tf.random.normal(tf.shape(x)) # (batch_size, *image_shape)
    time_index = np.random.choice(self.noise_level, (self.batch_size, ))
    variance = tf.gather(self.variance_schedule, time_index, axis = 0)
    broadcasted_variance = variance[:, tf.newaxis, tf.newaxis, tf.newaxis]
    broadcasted_noise = (broadcasted_variance ** 0.5) * epsilon
    x_corrupted = x + broadcasted_noise

    grad = (x - x_corrupted) / broadcasted_variance

    with tf.GradientTape() as tape:

      grad_pred = self.call(x_corrupted, time_index = time_index)

      loss = tf.reduce_sum(broadcasted_variance * (grad_pred - grad) ** 2)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.compiled_metrics.update_state(grad, grad_pred)
    return {**{m.name: m.result() for m in self.metrics}, "loss": loss}

  def langevin_dynamics(self, x, alpha, time_index, num_steps):
    if num_steps == 0:
      return x
    z_t = tf.random.normal(tf.shape(x))
    new_x = x + (alpha / 2.) * self(x, time_index) + (alpha ** 0.5) * z_t
    return self.langevin_dynamics(new_x, alpha, time_index, num_steps - 1)

  def annealed_langevin_dynamics(self, x, step_size=2e-5, num_steps = 100, return_intermediate=False):
    last_variance = self.variance_schedule[-1] # Smallest variance
    b = tf.shape(x)[0]
    x_t = x
    intermediate_samples = []
    for i in tqdm(range(self.noise_level)):
      variance_t = self.variance_schedule[i]
      alpha_t = step_size * variance_t / last_variance
      x_t = self.langevin_dynamics(x_t, alpha_t, tf.broadcast_to(i, (b,)), num_steps)
      intermediate_samples.append(x_t)

    if return_intermediate:
      return intermediate_samples
    return x_t
  

class UnguidedVideoDiffusion(tf.keras.models.Model):

  def __init__(self, input_shape = (64, 64),
               num_channels = 3,
               num_prev_frames = 4,
               batch_size = 32,
               start_filters = 256,
               num_blocks = 3,
               dropout_prob = 0,
               filter_size = 3,
               noise_level = 10,
               leaky = 0.05,
               use_action = False,
               **kwargs):

    super().__init__(**kwargs)
    self.total_channels = num_channels * (num_prev_frames + 1)
    self.unet = create_unet(
      input_shape + (self.total_channels, ),
      start_filters,
      num_blocks,
      dropout_prob,
      filter_size,
      leaky = leaky,
      output_channels = num_channels,
      use_action_embedding=use_action
      )
    
    self.unet.build([input_shape + (self.total_channels, ), (1, 1, start_filters), (1, 1, start_filters)])
    self.noise_level = noise_level
    self.image_input_shape = input_shape
    self.num_channels = num_channels
    self.num_prev_frames = num_prev_frames
    self.full_input_shape = (-1, ) + self.image_input_shape + (self.total_channels, )
    self.total_image_dims = self.image_input_shape[0] * self.image_input_shape[1] * self.total_channels
    self.batch_size = batch_size

    self.time_embeddings = []

    for block in range(num_blocks):
      time_embed = TimeEmbedding2D(start_filters, start_filters, name=f"Time_embedding_block_{block}")
      time_embed.build((1,))
      self.time_embeddings.append(
          time_embed
      )

    self.variance_schedule = self.linear_variance_schedule(noise_level)

  def linear_variance_schedule(self, timesteps):
    lin_start = 1.
    lin_end = 0.001
    return tf.cast(tf.linspace(lin_start, lin_end, timesteps), tf.float32)

  def call(self, x, time_index):

      # Time embedding
      time_context = [time_embed(time_index)[:, tf.newaxis, tf.newaxis, ...] for time_embed in self.time_embeddings]

      return self.unet([x, *time_context])

  def train_step(self, x, prev_frames):
    # Sample noise
    epsilon = tf.random.normal((self.batch_size, ) + self.image_input_shape + (self.num_channels, )) # (batch_size, *image_shape)
    time_index = np.random.choice(self.noise_level, (self.batch_size, ))
    variance = tf.gather(self.variance_schedule, time_index, axis = 0)
    broadcasted_variance = variance[:, tf.newaxis, tf.newaxis, tf.newaxis]
    broadcasted_noise = (broadcasted_variance ** 0.5) * epsilon
    x_corrupted = x + broadcasted_noise

    # Concatenate with non-corrupted previous frames
    frames = tf.concat([prev_frames, x_corrupted], axis = -1)

    grad = (x - x_corrupted) / broadcasted_variance

    with tf.GradientTape() as tape:

      grad_pred = self.call(frames, time_index = time_index)

      loss = tf.reduce_sum(broadcasted_variance * (grad_pred - grad) ** 2)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.compiled_metrics.update_state(grad, grad_pred)
    return {**{m.name: m.result() for m in self.metrics}, "loss": loss}

  def langevin_dynamics(self, x, alpha, time_index, num_steps, prev_frames = None):
    z_t = tf.random.normal(tf.shape(x))
    if prev_frames is not None:
      frames = tf.concat([prev_frames, x], axis = -1)
    else:
      frames = x
    if num_steps == 0:
      return x + (alpha / 2.) * self(frames, time_index)
    new_x = x + (alpha / 2.) * self(frames, time_index) + (alpha ** 0.5) * z_t
    return self.langevin_dynamics(new_x, alpha, time_index, num_steps - 1, prev_frames = prev_frames)

  def annealed_langevin_dynamics(self, x, step_size=2e-5, num_steps = 100, return_intermediate=False, prev_frames = None):
    last_variance = self.variance_schedule[-1] # Smallest variance
    b = tf.shape(x)[0]
    x_t = x
    intermediate_samples = []
    for i in tqdm(range(self.noise_level)):
      variance_t = self.variance_schedule[i]
      alpha_t = step_size * variance_t / last_variance
      x_t = self.langevin_dynamics(x_t, alpha_t, tf.broadcast_to(i, (b,)), num_steps, prev_frames)
      intermediate_samples.append(x_t)

    if return_intermediate:
      return intermediate_samples
    return x_t
  
  def sample_from_frames(self, frames, num_frames = 30, step_size=2e-5, num_steps = 100, batch_size = 32):
    new_frames = []

    print(tf.shape(frames))

    # Sample frame-by-frame
    for i in range(num_frames):
      assert frames.shape[-1] == self.num_prev_frames * self.num_channels, f"Frame-by-frame sampling failed on frame {i}"
      new_frame = self.annealed_langevin_dynamics(
        tf.random.normal(
          (batch_size, ) + self.image_input_shape + (self.num_channels, )),
        step_size = step_size,
        num_steps = num_steps,
        prev_frames = frames)
      
      frames = tf.concat([frames[..., self.num_channels:], new_frame], axis = -1)
      new_frames.append(new_frame)

    return tf.stack(new_frames, axis = 1) # Shape: (batch, frames, h, w, channels)
      
  def load_from_ckpt(self, path):
    if path is None:
      raise Exception("Trying to load from null path")
    
    self.call(
			tf.random.normal((self.batch_size, ) + self.full_input_shape[1:]),
			tf.random.normal((self.batch_size, )))
    self.load_weights(path)

class LatentActionVideoDiffusion(UnguidedVideoDiffusion):
  def __init__(self, input_shape = (64, 64),
               num_channels = 3,
               num_prev_frames = 4,
               batch_size = 32,
               start_filters = 256,
               num_blocks = 3,
               dropout_prob = 0,
               filter_size = 3,
               noise_level = 10,
               leaky = 0.05,
               regularized_lambda = VQVAE_LOSS_LAMBDA,
               gram_loss_lambda = GRAM_MATRIX_LAMBDA,
               start_step = True,
               **kwargs):
    # super(tf.keras.models.Model, self).__init__(**kwargs)
    UnguidedVideoDiffusion.__init__(self, input_shape = input_shape,
               num_channels = num_channels,
               num_prev_frames = num_prev_frames,
               batch_size = batch_size,
               start_filters = start_filters,
               num_blocks = num_blocks,
               dropout_prob = dropout_prob,
               filter_size = filter_size,
               noise_level = noise_level,
               leaky = leaky,
               use_action = True,
               **kwargs)
    # super(UnguidedVideoDiffusion, self).__init__(input_shape = input_shape,
    #            num_channels = num_channels,
    #            num_prev_frames = num_prev_frames,
    #            batch_size = batch_size,
    #            start_filters = start_filters,
    #            num_blocks = num_blocks,
    #            dropout_prob = dropout_prob,
    #            filter_size = filter_size,
    #            noise_level = noise_level,
    #            leaky = leaky,
    #            use_action = True,
    #            **kwargs)
    
    self.regularized_lambda = regularized_lambda
    self.gram_loss_lambda = gram_loss_lambda

    self.latent_action_model, self.latent_action_quantizer = get_image_vq_encoder(
      latent_dim=VQVAE_EMBEDDING_DIM,
      num_embeddings=VQVAE_NUM_EMBEDDINGS,
      image_shape=VQVAE_INPUT_SHAPE[:2],
      num_channels=VQVAE_INPUT_SHAPE[-1],
      ema=False,
      batchnorm=True
    )

    self.action_norm = tf.keras.layers.LayerNormalization()

    self.action_index_counter = tf.zeros((VQVAE_NUM_EMBEDDINGS, ))

    if start_step:
      self.step_count = tf.zeros((batch_size, ))
    else:
      self.step_count = (VQVAE_EXPLORE_STEPS + 1) * tf.ones((batch_size, ))

    self.create_encoder_model()
  
  def create_encoder_model(self, encoded_layer_name = "UNET_MIDDLE_BLOCK"):
    self.encoder_model = tf.keras.models.Model(inputs = self.unet.inputs[0], outputs = self.unet.get_layer(encoded_layer_name).output)

  def call_encoder_model(self, x):
    if not hasattr(self, "encoder_model"):
      print("No model detected: Creating Encoder Model... [calling self.create_encoder_model()]")
      self.create_encoder_model()
      print("Encoder Model created")
    return self.encoder_model(x)
  
  def set_action_embeddings(self):
    self.vq_action_embeddings = tf.transpose(self.latent_action_quantizer.embeddings)
  
  def call_outside_of_train(self, prev_frames, time_index, action_index):
    if not hasattr(self, "vq_action_embeddings"):
      print("Setting Action Embeddings from VQ layer")
      self.set_action_embeddings()
    quantized_action = tf.nn.embedding_lookup(self.vq_action_embeddings, action_index)

    

    return self.call(prev_frames, time_index, quantized_action[:, tf.newaxis, tf.newaxis, ...])
  
  def call(self, prev_frames, time_index, quantized_action_embedding):

    # quantized_embeddings = quantized_action_embedding[:, tf.newaxis, tf.newaxis, ...]
      
    # quantized_embeddings = tf.nn.embedding_lookup(self.latent_action_quantizer.embeddings, action_index)[:, tf.newaxis, tf.newaxis, ...]

    # Time embedding
    time_context = [time_embed(time_index)[:, tf.newaxis, tf.newaxis, ...] for time_embed in self.time_embeddings]

    return self.unet([prev_frames, quantized_action_embedding, *time_context])
  
  def gram_matrix_loss(self):
    # Taken from https://www.tensorflow.org/tutorials/generative/style_transfer#calculate_style
    normed_embeds = tf.math.l2_normalize(self.latent_action_quantizer.embeddings, axis = 0)
    result = tf.reduce_sum(tf.transpose(normed_embeds) @ normed_embeds)
    num_locations = tf.cast(VQVAE_NUM_EMBEDDINGS*VQVAE_EMBEDDING_DIM, tf.float32)
    return result/(num_locations)
  
  def train_step(self, x, prev_frames, future_frames):
    # Sample noise
    epsilon = tf.random.normal((self.batch_size, ) + self.image_input_shape + (self.num_channels, )) # (batch_size, *image_shape)
    time_index = np.random.choice(self.noise_level, (self.batch_size, ))
    variance = tf.gather(self.variance_schedule, time_index, axis = 0)
    broadcasted_variance = variance[:, tf.newaxis, tf.newaxis, tf.newaxis]
    broadcasted_noise = (broadcasted_variance ** 0.5) * epsilon
    x_corrupted = x + broadcasted_noise

    # Concatenate with non-corrupted previous frames
    frames = tf.concat([prev_frames, x_corrupted], axis = -1)

    grad = (x - x_corrupted) / broadcasted_variance

    future_encoding = self.call_encoder_model(future_frames)
    prev_encoding = self.call_encoder_model(tf.concat([prev_frames, tf.zeros_like(x_corrupted, dtype = tf.float32)], axis = -1))

    encoding_diff_unnormalized = future_encoding - prev_encoding

    with tf.GradientTape() as tape:   

      encoding_diff = self.action_norm(encoding_diff_unnormalized)

      quantized_action, original_encoding_indices = self.latent_action_model([encoding_diff, self.step_count])

      # print(tf.shape(quantized_action))

      grad_pred = self.call(frames, time_index = time_index, quantized_action_embedding = quantized_action)

      # VQ regularization losses
      vq_loss = sum(self.losses)

      # Generation loss
      diffusion_loss = tf.reduce_mean(broadcasted_variance * (grad_pred - grad) ** 2)

      # Loss to make every action embedding orthogonal to each other
      # gram_loss = self.gram_matrix_loss()

      loss = diffusion_loss + self.regularized_lambda * vq_loss # + self.gram_loss_lambda * gram_loss

    # Add action indices to counter
      
    self.action_index_counter += tf.reduce_sum(tf.one_hot(original_encoding_indices, VQVAE_NUM_EMBEDDINGS), axis = 0)

    # Loss to make every action embedding orthogonal to each other
    gram_loss = self.gram_matrix_loss()

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    self.compiled_metrics.update_state(grad, grad_pred)
    self.step_count = self.step_count + 1.
    return {**{m.name: m.result() for m in self.metrics}, "loss": loss, "vq_loss (unscaled term)": vq_loss, "diffusion_loss": diffusion_loss, "gram_loss": gram_loss}
  
  def langevin_dynamics(self, x, alpha, time_index, action_index, num_steps, prev_frames = None, lamb = CONDITIONAL_SAMPLING_LAMBDA):
    z_t = tf.random.normal(tf.shape(x))
    if prev_frames is not None:
      frames = tf.concat([prev_frames, x], axis = -1)
    else:
      frames = x

    grad_p_x = self.call_outside_of_train(frames, time_index, tf.zeros_like(action_index))
    grad_p_x_y = self.call_outside_of_train(frames, time_index, action_index)

    # The central equation for classifier-free guidance
    grad_p = (1 - lamb) * grad_p_x + lamb * grad_p_x_y

    new_x = x + (alpha / 2.) * grad_p + (alpha ** 0.5) * z_t
    if num_steps == 0:
      return new_x
    return self.langevin_dynamics(new_x, alpha, time_index, action_index, num_steps - 1, prev_frames = prev_frames, lamb = lamb)

  def annealed_langevin_dynamics(self, x, action_index, step_size=2e-5, num_steps = 100, return_intermediate=False, prev_frames = None, lamb = CONDITIONAL_SAMPLING_LAMBDA):
    last_variance = self.variance_schedule[-1] # Smallest variance
    b = tf.shape(x)[0]
    x_t = x
    intermediate_samples = []
    for i in tqdm(range(self.noise_level)):
      variance_t = self.variance_schedule[i]
      alpha_t = step_size * variance_t / last_variance
      x_t = self.langevin_dynamics(x_t, alpha_t, tf.broadcast_to(i, (b,)), action_index, num_steps, prev_frames, lamb)
      intermediate_samples.append(x_t)

    if return_intermediate:
      return intermediate_samples
    return x_t
  
  def sample_from_frames(self, frames, action_index, num_frames = 30, step_size=2e-5, num_steps = 100, batch_size = 32, lamb = CONDITIONAL_SAMPLING_LAMBDA):
    new_frames = []

    print(tf.shape(frames))

    # Sample frame-by-frame
    for i in range(num_frames):
      assert frames.shape[-1] == self.num_prev_frames * self.num_channels, f"Frame-by-frame sampling failed on frame {i}"
      new_frame = self.annealed_langevin_dynamics(
        tf.random.normal(
          (batch_size, ) + self.image_input_shape + (self.num_channels, )),
        action_index = action_index,
        step_size = step_size,
        num_steps = num_steps,
        prev_frames = frames,
        lamb = lamb)
      
      frames = tf.concat([frames[..., self.num_channels:], new_frame], axis = -1)
      new_frames.append(new_frame)

    return tf.stack(new_frames, axis = 1) # Shape: (batch, frames, h, w, channels)
  
  def load_from_ckpt(self, path):
    if path is None:
      raise Exception("Trying to load from null path")
    
    self.call(
			prev_frames = tf.random.normal((self.batch_size, ) + self.full_input_shape[1:]),
			time_index = tf.random.normal((self.batch_size, )),
      quantized_action_embedding = tf.random.normal((self.batch_size, 1, 1, VQVAE_EMBEDDING_DIM)))
    self.load_weights(path)