from imports import tf, np

preprocess_standard = lambda im : (im / 127.) - 1.0

# Data-gen
def create_flow_unguided(x, batch_size = 128, preprocess_func = None, repeat = True):
  i = 0
  total_samples = x.shape[0]
  while True:
    i += batch_size
    if ((i + batch_size) >= total_samples) and repeat:
      i = 0
      idxs = tf.range(total_samples, dtype = tf.int32)
      shuffled_idxs = tf.random.shuffle(idxs)
      x = tf.gather(x, shuffled_idxs)
    elif (i + batch_size) >= total_samples:
      if preprocess_func is None:
        yield x[i: i + batch_size]
      else:
        yield preprocess_func(x[i: i + batch_size])
      break
    if preprocess_func is None:
      yield x[i: i + batch_size]
    else:
      yield preprocess_func(x[i: i + batch_size])

# Load Latent Data
def load_latent_data(fp_to_npy, preprocess_func = None):
  if type(fp_to_npy) == str:
    fp_to_npy = [fp_to_npy]
  
  results_list = []
  runningFrameCount = 0
  episode_changes = []
  for fp in fp_to_npy:
    with open(fp, "rb") as npy_file:
      results = np.load(npy_file)
    results_list.append(
      preprocess_func(results) if preprocess_func is not None else results
      )
    runningFrameCount += results.shape[0]
    episode_changes.append(runningFrameCount)

  print(f"Total frame count: {runningFrameCount}")
  print(f"Episode-changing frames: {episode_changes}")
  return np.concatenate(results_list, axis = 0), episode_changes

def calc_frame_indices(total_samples, num_frames_per_sample, episode_changes = []):
  total_sample_indices = [i for i in range(total_samples)]
  available_sample_indices = total_samples - num_frames_per_sample
  stacked_indices = []
  for i in range(available_sample_indices):
    continueFlag = False
    for episode_change in episode_changes:
      if (episode_change - num_frames_per_sample) <= i <= episode_change:
        continueFlag = True
    if continueFlag:
      continue
    stacked_indices.append((
      total_sample_indices[i:i+num_frames_per_sample]
    ))
  return np.array(stacked_indices)

def calc_frame_indices_with_future_frames(total_samples, num_frames_per_sample, future_offset, episode_changes = []):
  total_sample_indices = [i for i in range(total_samples)]
  available_sample_indices = total_samples - num_frames_per_sample - future_offset
  stacked_indices = []
  stacked_future_indices = []
  for i in range(available_sample_indices):
    continueFlag = False
    for episode_change in episode_changes:
      if (episode_change - num_frames_per_sample - future_offset) <= i <= episode_change:
        continueFlag = True
    if continueFlag:
      continue
    stacked_indices.append((
      total_sample_indices[i:i+num_frames_per_sample]
    ))
    stacked_future_indices.append(
      total_sample_indices[i+future_offset:i+num_frames_per_sample+future_offset]
    )
  return np.array(stacked_indices), np.array(stacked_future_indices)

def gather_samples_from_dataset(X_y_indices, dataset):
  # X_indices = []
  # y_indices = []

  # for X_ind, y_ind in X_y_indices:
  #   X_indices.append(X_ind)
  #   y_indices.append(y_ind)

  X = tf.gather(dataset, X_y_indices[:, :-1], axis = 0)
  y = tf.gather(dataset, X_y_indices[:, -1], axis = 0)
  return X, y