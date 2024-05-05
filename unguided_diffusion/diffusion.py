from constants import VQVAE_EMBEDDING_DIM, VQVAE_NUM_EMBEDDINGS, VQVAE_LOSS_LAMBDA, CONDITIONAL_SAMPLING_LAMBDA, GRAM_MATRIX_LAMBDA, VQVAE_EXPLORE_STEPS
from imports import tf, np, tqdm
from unguided_diffusion.unet import create_unet
from unguided_diffusion.model_blocks import TimeEmbedding2D
from vqvae.model import ImageVQEncoder

########### UNGUIDED VIDEO DIFFUSION MODEL ###########

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
    """
    Unguided Video Diffusion model that uses UNet as the internal architecture and frame-stacking to process video footage.

    Inputs:
    - input_shape | Tuple[int, int] : input image height and width.
    - num_channels | int : input number of channels.
    - num_prev_frames | int : number of previous frames used to generate next frame.
    - batch_size | int : input batch size.
    - start_filters | int : number of filters for all convolutional layer in UNet.
    - num_blocks | int : number of encoder and decoder blocks in the UNet. Each encoder/decoder block downsamples/upsamples the inputs by a scale 2.
    - dropout_prob | float : dropout percentage for dropout layer in each encoder/decoder block.
    - filter_size | int : kernel size for each convolutional layer in UNet.
    - noise_level | int : number of noise levels for diffusion generation.
    - leaky | float : percentage leakage for leaky relu layers.
    - use_action | bool : whether this model takes in action embeddings or not. Default is False, usually only use True through LatentActionVideoDiffusion model.
    """

    super().__init__(**kwargs)

    # Based on frame-stacking, we stack all the previous frames on the channel axis.
    # + 1 refers to the current generating frame.
    self.total_channels = num_channels * (num_prev_frames + 1)

    # Initialize UNet
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

    # In order to condition model on noise level, we have time embedding for every decoder block in UNet.
    # Input shape for each embedding is (1, ) being the noise level t.
    # Output shape is (1, 1, 512) to be broadcastable and concat-able to any decoder layer input.
    self.time_embeddings = []
    for block in range(num_blocks):
      time_embed = TimeEmbedding2D(start_filters, start_filters, name=f"Time_embedding_block_{block}")
      time_embed.build((1,))
      self.time_embeddings.append(
          time_embed
      )

    self.variance_schedule = self.linear_variance_schedule(noise_level)

  def linear_variance_schedule(self, timesteps):
    """
    Create an inverse linear variance schedule for the noise levels.

    Input:
    - timesteps | int : number of unique noise levels.

    Output:
    - tf.tensor[float] : array of floats representing variance for each noise level.
    """
    lin_start = 1.
    lin_end = 0.001
    return tf.cast(tf.linspace(lin_start, lin_end, timesteps), tf.float32)

  def call(self, x, time_index):
      """
      Call function, overrode from tf.keras.Model class.
      Calls the internal UNet to calculate $grad log p(x_t)$.

      Inputs:
      - x | tf.tensor : frame-stacked input with previous frames and current noisy frame. Shape (b, h, w, c * (num_prev_frames + 1))
      - time_index | int : Current noise level.

      Output:
      - tf.tensor : noise to be removed from x_t to move towards x_{t+1} with lower noise level.
      """

      # Time embedding
      time_context = [time_embed(time_index)[:, tf.newaxis, tf.newaxis, ...] for time_embed in self.time_embeddings]

      return self.unet([x, *time_context])

  def train_step(self, x, prev_frames):
    """
    Train step function, overrode from tf.keras.Model class.
    Trains model for one step. We use a weighted mean squared error loss to train the model.

    Inputs:
    - x | tf.tensor : the current frame that the model wants to learn to generate. Shape (b, h, w, c)
    - prev_frames | tf.tensor : the previous frames as context for the model to generate the current frame. Shape (b, h, w, c * num_prev_frames)

    Output:
    - Dict[str, float] : metrics calculated based on current train step iteration. Includes loss.
    """
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
    """
    Langevin Dynamics sampling at a particular noise level.

    Inputs:
    - x | tf.tensor : noisy current frame to be generated/sampled.
    - alpha | float : time-conditioned constant to scale sampling strength.
    - time_index | int : current time index / noise level.
    - num_steps | int : number of steps left for the sampling at this noise level.
    - prev_frames | tf.tensor : previous frames to be used as context to generate current frame.

    Outputs:
    - tf.tensor : x which has been sampled num_steps at the current noise level with Langevin Dynamics.
    """
    z_t = tf.random.normal(tf.shape(x))
    if prev_frames is not None:
      frames = tf.concat([prev_frames, x], axis = -1)
    else:
      frames = x
    if num_steps == 0:
      return x + (alpha / 2.) * self(frames, time_index)
    
    # Generate x_{t+1} with x_t using the time-conditioned constant alpha
    # as well as some stochastic gaussian noise to ensure non-deterministic results.
    new_x = x + (alpha / 2.) * self(frames, time_index) + (alpha ** 0.5) * z_t
    return self.langevin_dynamics(new_x, alpha, time_index, num_steps - 1, prev_frames = prev_frames)

  def annealed_langevin_dynamics(self, x, step_size=2e-5, num_steps = 100, return_intermediate=False, prev_frames = None):
    """
    Perform Annealed Langevin Dynamics sampling on some noisy input image x.
    See Algorithm 1 in https://arxiv.org/pdf/1907.05600 for implementation details.

    Inputs:
    - x | tf.tensor : noisy current frame to be generated/sampled.
    - step_size | float : constant to determine the step size for every Langevin Dynamics sampling step.
    - num_steps | int : number of steps to sample per noise level.
    - return_intermediate | bool : whether to return intermediate generation samples for each noise level result.
    - prev_frames | tf.tensor : previous frames to be used as context to generate current frame.

    Outputs:
    - Union[List[tf.tensor], tf.tensor] : list of samples generated from every noise level or the final generated samples at the last noise level.
    """
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
    """
    Continuously samples num_frames number of frames given some starting frames.
    Essentially, this generates a num_frames video because we use the newly generated frame to generate the next frame.

    Inputs:
    - frames | tf.tensor : previous frames to generate next frame from.
    - num_frames | int : number of frames to generate.
    - step_size | float : constant to determine the step size for every Langevin Dynamics sampling step.
    - num_steps | int : number of steps per noise level to perform Langevin Dynamics.
    - batch_size | int : number of concurrent samples to generate.

    Outputs:
    - tf.tensor : tensor of newly generated frames.
    """
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
    """
    Loads weights from checkpoint to self model. Uses Tensorflow .index checkpoints.

    Inputs: 
    - path | str : path to checkpoint
    """
    if path is None:
      raise Exception("Trying to load from null path")
    
    self.call(
			tf.random.normal((self.batch_size, ) + self.full_input_shape[1:]),
			tf.random.normal((self.batch_size, )))
    self.load_weights(path)


####################################################################################################################################
####################################################################################################################################


########### LATENT ACTION VIDEO DIFFUSION MODEL ###########

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
    """
    Latent Action Video Diffusion Model that inherits from the Unguided Video Diffusion model.
    Unguided Video Diffusion model that uses UNet as the internal architecture and frame-stacking to process video footage.

    Inputs:
    - input_shape | Tuple[int, int] : input image height and width.
    - num_channels | int : input number of channels.
    - num_prev_frames | int : number of previous frames used to generate next frame.
    - batch_size | int : input batch size.
    - start_filters | int : number of filters for all convolutional layer in UNet.
    - num_blocks | int : number of encoder and decoder blocks in the UNet. Each encoder/decoder block downsamples/upsamples the inputs by a scale 2.
    - dropout_prob | float : dropout percentage for dropout layer in each encoder/decoder block.
    - filter_size | int : kernel size for each convolutional layer in UNet.
    - noise_level | int : number of noise levels for diffusion generation.
    - leaky | float : percentage leakage for leaky relu layers.
    - regularized_lambda | float : lambda for scaling the Vector Quantizer loss terms (codebook and commitment).
    - (DEPRICATED) gram_loss_lambda | float : lambda for scaling the gram matrix loss term, no longer using gram matrix loss.
    """

    # Inherit all properties from UnguidedVideoDiffusion class
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
    
    self.regularized_lambda = regularized_lambda
    self.gram_loss_lambda = gram_loss_lambda

    # Instantiate Latent Action model as a Vector Quantized Encoder
    self.latent_action_model = ImageVQEncoder(latent_dim=VQVAE_EMBEDDING_DIM,
      num_embeddings=VQVAE_NUM_EMBEDDINGS,
      ema=False,
      batchnorm=True)
    
    # Extract the Vector Quantizer layer from the latent action model
    # for direct access to the action embeddings later.
    self.latent_action_quantizer = self.latent_action_model.get_vq_layer()

    # For normalizing Latent Action model inputs
    self.action_norm = tf.keras.layers.LayerNormalization()

    # To keep track of how many times each action is used during training
    # and make sure that the model does not suffer from index collapse.
    self.action_index_counter = tf.zeros((VQVAE_NUM_EMBEDDINGS, ))

    if start_step:
      self.step_count = tf.zeros((batch_size, ))
    else:
      self.step_count = (VQVAE_EXPLORE_STEPS + 1) * tf.ones((batch_size, ))

    self.create_encoder_model()
  
  def create_encoder_model(self, encoded_layer_name = "UNET_MIDDLE_BLOCK"):
    """
    Extracts a pointer to the UNet encoder.

    Input:
    - encoded_layer_name | str : name of UNet encoder output layer.
    """
    self.encoder_model = tf.keras.models.Model(inputs = self.unet.inputs[0], outputs = self.unet.get_layer(encoded_layer_name).output)

  def call_encoder_model(self, x):
    """
    Calls the extracted UNet encoder.

    Input:
    - x | tf.tensor : input image for UNet encoder.

    Output:
    - tf.tensor : Encoded UNet encoder output.
    """
    if not hasattr(self, "encoder_model"):
      print("No model detected: Creating Encoder Model... [calling self.create_encoder_model()]")
      self.create_encoder_model()
      print("Encoder Model created")
    return self.encoder_model(x)
  
  def set_action_embeddings(self):
    """
    Extracts action embeddings from Latent Action Vector Quantizer.
    """
    self.vq_action_embeddings = tf.transpose(self.latent_action_quantizer.embeddings)
  
  def call_outside_of_train(self, prev_frames, time_index, action_index):
    """
    Call UNet model outside of training sequence.
    Uses action index to condition UNet to predict $grad log p(x_t | y)$
    where the action embedding is the conditioning variable y.

    Inputs:
    - prev_frames | tf.tensor : frames including previous and current frames frame-stacked.
    - time_index | tf.tensor : current noise level for each sample in the batch.
    - action_index | tf.tensor : action to condition sampling on for each sample in the batch.
    """
    if not hasattr(self, "vq_action_embeddings"):
      print("Setting Action Embeddings from VQ layer")
      self.set_action_embeddings()
    quantized_action = tf.nn.embedding_lookup(self.vq_action_embeddings, action_index)

    return self.call(prev_frames, time_index, quantized_action[:, tf.newaxis, tf.newaxis, ...])
  
  def call(self, prev_frames, time_index, quantized_action_embedding):
    """
    Call function, overrode from UnguidedVideoDiffusion.
    Calls the internal UNet with action embeddings.

    Inputs:
    - prev_frames | tf.tensor : frames including previous and current frames frame-stacked.
    - time_index | tf.tensor : current noise level for each sample in the batch.
    - quantized_action_embedding | tf.tensor : action embeddings quantized with the Vector Quantizer.
    """
    # Time embedding
    time_context = [time_embed(time_index)[:, tf.newaxis, tf.newaxis, ...] for time_embed in self.time_embeddings]

    return self.unet([prev_frames, quantized_action_embedding, *time_context])
  
  def gram_matrix_loss(self):
    """
    Gram Matrix Metric, depricated loss term. Calculates the internal Cosine Similarity between action embeddings.

    Outputs:
    - tf.tensor : cosine similarity between action embeddings.
    """
    # Taken from https://www.tensorflow.org/tutorials/generative/style_transfer#calculate_style
    normed_embeds = tf.math.l2_normalize(self.latent_action_quantizer.embeddings, axis = 0)
    result = tf.reduce_sum(tf.transpose(normed_embeds) @ normed_embeds)
    num_locations = tf.cast(VQVAE_NUM_EMBEDDINGS*VQVAE_EMBEDDING_DIM, tf.float32)
    return result/(num_locations)
  
  def train_step(self, x, prev_frames, future_frames):
    """
    Train step function, overrode from tf.keras.Model class.
    Trains model for one step.

    Inputs:
    - x | tf.tensor : the current frame that the model wants to learn to generate. Shape (b, h, w, c)
    - prev_frames | tf.tensor : the previous frames as context for the model to generate the current frame. Shape (b, h, w, c * num_prev_frames)
    - future_frames | tf.tensor : the future frames as data-leaking context to find the best quantized action embedding. Shape (b, h, w, c * (num_prev_frames + 1))

    Output:
    - Dict[str, float] : metrics calculated based on current train step iteration. Includes loss, diffusion loss, Vector Quantizer loss, and Gram Matrix loss.
    """
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

    # Calculate the encoded difference between the future and previous frames
    # to be encoded and quantized into the "best course of action".
    # Note: No gradient for encoder model here, may be an area for future exploration.
    future_encoding = self.call_encoder_model(future_frames)
    prev_encoding = self.call_encoder_model(tf.concat([prev_frames, tf.zeros_like(x_corrupted, dtype = tf.float32)], axis = -1))

    latent_diff_unnormalized = future_encoding - prev_encoding

    with tf.GradientTape() as tape:   

      # Calculate quantized action based on leaked future frames
      latent_diff = self.action_norm(latent_diff_unnormalized)
      quantized_action, original_encoding_indices = self.latent_action_model(latent_diff, self.step_count)

      # Predict gradient of log prob x given action
      grad_pred = self.call(frames, time_index = time_index, quantized_action_embedding = quantized_action)

      # VQ regularization losses
      vq_loss = sum(self.losses)

      # Generation loss
      diffusion_loss = tf.reduce_mean(broadcasted_variance * (grad_pred - grad) ** 2)

      # (DEPRICATED) Loss to make every action embedding orthogonal to each other
      # gram_loss = self.gram_matrix_loss()

      # Weighted sum all losses
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
    """
    Langevin Dynamics sampling at a particular noise level with Classifier-Free Guidance.

    Inputs:
    - x | tf.tensor : noisy current frame to be generated/sampled.
    - alpha | float : time-conditioned constant to scale sampling strength.
    - time_index | int : current time index / noise level.
    - num_steps | int : number of steps left for the sampling at this noise level.
    - prev_frames | tf.tensor : previous frames to be used as context to generate current frame.

    Outputs:
    - tf.tensor : x which has been sampled num_steps at the current noise level with Langevin Dynamics.
    """
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
    """
    Perform Annealed Langevin Dynamics sampling on some noisy input image x.
    See Algorithm 1 in https://arxiv.org/pdf/1907.05600 for implementation details.

    Inputs:
    - x | tf.tensor : noisy current frame to be generated/sampled.
    - step_size | float : constant to determine the step size for every Langevin Dynamics sampling step.
    - num_steps | int : number of steps to sample per noise level.
    - return_intermediate | bool : whether to return intermediate generation samples for each noise level result.
    - prev_frames | tf.tensor : previous frames to be used as context to generate current frame.

    Outputs:
    - Union[List[tf.tensor], tf.tensor] : list of samples generated from every noise level or the final generated samples at the last noise level.
    """
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
    """
    Continuously samples num_frames number of frames given some starting frames.
    Essentially, this generates a num_frames video because we use the newly generated frame to generate the next frame.

    Inputs:
    - frames | tf.tensor : previous frames to generate next frame from.
    - num_frames | int : number of frames to generate.
    - step_size | float : constant to determine the step size for every Langevin Dynamics sampling step.
    - num_steps | int : number of steps per noise level to perform Langevin Dynamics.
    - batch_size | int : number of concurrent samples to generate.

    Outputs:
    - tf.tensor : tensor of newly generated frames.
    """
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
    """
    Loads weights from checkpoint to self model. Uses Tensorflow .index checkpoints.

    Inputs: 
    - path | str : path to checkpoint
    """
    if path is None:
      raise Exception("Trying to load from null path")
    
    self.call(
			prev_frames = tf.random.normal((self.batch_size, ) + self.full_input_shape[1:]),
			time_index = tf.random.normal((self.batch_size, )),
      quantized_action_embedding = tf.random.normal((self.batch_size, 1, 1, VQVAE_EMBEDDING_DIM)))
    self.load_weights(path)