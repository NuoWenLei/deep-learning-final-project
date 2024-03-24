from imports import tf, np, tqdm

from unguided_diffusion.helpers import create_flow_unguided, gather_samples_from_dataset, load_latent_data, calc_frame_indices
from unguided_diffusion.diffusion import UnguidedVideoDiffusion

import gc
import os
import json
import datetime
from functools import partial
from collections import defaultdict

from emailSender import send_attached_email, send_email
from constants import (
	# Training Settings
	LOG_FILEPATH,
	RESULT_DIRPATH,
	RESULT_SAMPLE_RATE,
	SAMPLE_BATCH_SIZE,
	LATENT_SAMPLE_PATH,
	CHECKPOINT_PATH,
	CHECKPOINT_SAVE_RATE,
	BATCH_SIZE,
	NUM_EPOCHS,
	USE_EMAIL_NOTIFICATION,
	USE_SAMPLE_DATA,
	
	# Model Params
	LATENT_SHAPE,
	NUM_FILTERS,
	NUM_BLOCKS,
	NUM_PREV_FRAMES,
	DROPOUT_PROB,
	FILTER_SIZE,
	NOISE_LEVELS,
	LEAKY,
	
	# Optimizer
	LEARNING_RATE)


def log(msg, filepath):
	if not os.path.isfile(filepath):
		with open(filepath, "w") as f:
			f.write("")

	with open(filepath, "a") as f:
		datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		f.write(f"\n{datetimeStr}: \n {msg}\n")
	return msg

def main():

	logger = partial(log, filepath=LOG_FILEPATH)

	if USE_SAMPLE_DATA:
		sample_latents = load_latent_data(LATENT_SAMPLE_PATH)
		NUM_SAMPLES = sample_latents.shape[0]
		sample_indices = calc_frame_indices(NUM_SAMPLES, NUM_PREV_FRAMES + 1)

		assert len(sample_indices) <= NUM_SAMPLES

		gather_func = partial(gather_samples_from_dataset, dataset = sample_latents)

		# Create Data Generator
		dataloader = create_flow_unguided(sample_indices, batch_size = BATCH_SIZE, preprocess_func=gather_func)
		test_dataloader = create_flow_unguided(sample_indices, batch_size = SAMPLE_BATCH_SIZE, preprocess_func=gather_func)
		logger("USE_SAMPLE_DATA=True, using sample data")
		logger(f"Sample indices shape: {sample_indices.shape}")

	# Initialize Diffusion Model
	diffusion_model = UnguidedVideoDiffusion(
		input_shape=LATENT_SHAPE[:2],
		num_channels=LATENT_SHAPE[-1],
		num_prev_frames=NUM_PREV_FRAMES,
		batch_size = BATCH_SIZE,
		start_filters=NUM_FILTERS,
		num_blocks=NUM_BLOCKS,
		dropout_prob=DROPOUT_PROB,
		filter_size=FILTER_SIZE,
		noise_level=NOISE_LEVELS,
		leaky=LEAKY,
		name="unguided_video_diffusion"
	)

	diffusion_model.compile(
			loss = "mse",
			optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE),
			metrics = ["mse"]
	)

	logger("Model compiled")

	STEPS_PER_EPOCH = NUM_SAMPLES // BATCH_SIZE

	summed_metrics = defaultdict(lambda: 0)

	for epoch in range(NUM_EPOCHS):
		logger(f"Epoch {epoch}: ")
		pb = tf.keras.utils.Progbar(BATCH_SIZE * STEPS_PER_EPOCH)
		for _ in range(STEPS_PER_EPOCH):
			# Sample next batch of data
			prev_frames_batch, new_frame_batch = next(dataloader)

			# Train and update metrics
			metrics = diffusion_model.train_step(new_frame_batch, prev_frames_batch)
			pb.add(BATCH_SIZE, values=[(k, v) for k,v in metrics.items()])

		# Calculate Metric Averages of Epoch
		epoch_averages = dict((k, v[0] / v[1]) for k, v in pb._values.items())
		for k in epoch_averages.keys():
			summed_metrics[k] += epoch_averages[k]

		if (CHECKPOINT_SAVE_RATE is not None) and (epoch % CHECKPOINT_SAVE_RATE == 0):
			save_path = os.path.join(CHECKPOINT_PATH, f"e{epoch}.h5")
			diffusion_model.save_weights(save_path)

			if USE_EMAIL_NOTIFICATION:

				send_email(f"""
								Reached checkpoint at Epoch {epoch}!
								\n\n\n
								Epoch Averages: {str(epoch_averages)}
								""")
			else:
				print(f"""
								Reached checkpoint at Epoch {epoch}!
								\n\n\n
								Epoch Averages: {str(epoch_averages)}
								""")

		if (RESULT_SAMPLE_RATE is not None) and (epoch % RESULT_SAMPLE_RATE == 0):
			prev_frames_sample_batch, _ = next(test_dataloader)
			new_frames = diffusion_model.sample_from_frames(
				prev_frames_sample_batch,
				num_frames = SAMPLE_BATCH_SIZE,
				batch_size = SAMPLE_BATCH_SIZE).numpy()
			
			# Min-max scale and transform to uint8 for storage efficiency
			new_frames_min_shift = (new_frames - new_frames.min())
			new_frames_uint = ((new_frames_min_shift / new_frames_min_shift.max()) * 255.).astype("uint8")

			# Save sample to results directory
			save_path = os.path.join(RESULT_DIRPATH, f"e{epoch}_sample.npy")
			with open(save_path, "wb") as npy_file:
				np.save(npy_file, new_frames_uint)

			# TODO: turn sample into video and send email
				
	if (CHECKPOINT_SAVE_RATE is None) or (epoch % CHECKPOINT_SAVE_RATE != 0):
		save_path = os.path.join(CHECKPOINT_PATH, f"e{epoch}.h5")
		diffusion_model.save_weights(save_path)


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		if USE_EMAIL_NOTIFICATION:
			send_email(f"Process exited with an error: {str(e)}")
		else:
			print(f"Process exited with an error: {str(e)}")