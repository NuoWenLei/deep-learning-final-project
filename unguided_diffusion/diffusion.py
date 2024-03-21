from imports import tf, np, tqdm
from unguided_diffusion.unet import create_unet
from unguided_diffusion.model_blocks import TimeEmbedding2D

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
      output_channels = num_channels)
    
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
    if num_steps == 0:
      return x
    z_t = tf.random.normal(tf.shape(x))
    if prev_frames is not None:
      frames = tf.concat([prev_frames, x], axis = -1)
    else:
      frames = x
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
