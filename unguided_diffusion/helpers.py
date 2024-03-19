from imports import tf

# Data-gen
def create_flow_unguided(x, batch_size = 128):
  i = 0
  total_samples = x.shape[0]
  while True:
    i += batch_size
    if (i + batch_size) >= total_samples:
      i = 0
      idxs = tf.range(total_samples, dtype = tf.int32)
      shuffled_idxs = tf.random.shuffle(idxs)
      x = tf.gather(x, shuffled_idxs)
    yield (tf.cast(x[i: i + batch_size], tf.float32) / 255.)