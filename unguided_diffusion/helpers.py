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
        yield (tf.cast(x[i: i + batch_size], tf.float32) / 255.)
      else:
        yield preprocess_func(tf.cast(x[i: i + batch_size], tf.float32))
      break
    if preprocess_func is None:
      yield (tf.cast(x[i: i + batch_size], tf.float32) / 255.)
    else:
      yield preprocess_func(tf.cast(x[i: i + batch_size], tf.float32))

# Load Latent Data
def load_latent_data(fp_to_npy, preprocess_func = None):
  with open(fp_to_npy, "rb") as npy_file:
    results = np.load(npy_file)

  if preprocess_func is not None:
    return preprocess_func(results)
  return results