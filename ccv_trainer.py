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
	EXPLODING_LOSS_DETECTION,
	BASE_FILEPATH,
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
	
	if not USE_EMAIL_NOTIFICATION:
		print(msg)
	return msg

def main(path_to_checkpoint = None):
	print("Starting CCV Trainer")
	logger = partial(log, filepath=os.path.join(BASE_FILEPATH, LOG_FILEPATH))

	if USE_SAMPLE_DATA:
		print("Using Sample Data")
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
	if path_to_checkpoint is not None:
		diffusion_model.load_from_ckpt(path_to_checkpoint)
		logger(f"Checkpoint loaded from: {path_to_checkpoint}")

	logger("Model compiled")

	STEPS_PER_EPOCH = NUM_SAMPLES // BATCH_SIZE

	# halfway_steps = STEPS_PER_EPOCH // 2

	summed_metrics = defaultdict(lambda: 0)
	prev_epoch_averages = None
	latest_save_path = None

	for epoch in range(NUM_EPOCHS):
		logger(f"Epoch {epoch}, Num Steps {STEPS_PER_EPOCH}: ")
		pb = tf.keras.utils.Progbar(BATCH_SIZE * STEPS_PER_EPOCH)
		for step in range(STEPS_PER_EPOCH):

			# if step % step_checkmarks == 0:
			# 	logger(f"Step {step}:\n\nCurrent epoch averages: {str(dict((k, v[0] / v[1]) for k, v in pb._values.items()))}")
			# Sample next batch of data
			prev_frames_batch, new_frame_batch = next(dataloader)

			prev_frames_reshaped = tf.reshape(
				tf.transpose(
					prev_frames_batch,
					[0, 2, 3, 1, 4]),
					(BATCH_SIZE, ) + LATENT_SHAPE[:-1] + (-1, ))

			# Train and update metrics
			metrics = diffusion_model.train_step(new_frame_batch, prev_frames_reshaped)
			pb.add(BATCH_SIZE, values=[(k, v) for k,v in metrics.items()])

		# Calculate Metric Averages of Epoch
		epoch_averages = dict((k, v[0] / v[1]) for k, v in pb._values.items())
		for k in epoch_averages.keys():
			summed_metrics[k] += epoch_averages[k]

		if prev_epoch_averages is not None:
			# Check if there is exploding gradient by detecting abnormally high loss
			if epoch_averages["loss"] > (EXPLODING_LOSS_DETECTION * prev_epoch_averages["loss"]):
				msg = logger(f"""
						Exploding Gradient detected with following metrics:
						
						Previous Epoch Averages
						{prev_epoch_averages}

						Current Epoch Averages
						{epoch_averages}
						
						Attempting to restore latest save
						""")
				
				if USE_EMAIL_NOTIFICATION:
					send_email(msg)
				
				if (latest_save_path is None) and (path_to_checkpoint is None):
					send_email(logger("No previous save to restore, raising error"))
					raise Exception("No previous save to restore from exploding gradient")
				else:
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
					restored_path = latest_save_path if latest_save_path is not None else path_to_checkpoint
					diffusion_model.load_from_ckpt(restored_path)

					msg = logger(f"Latest save restore from: {restored_path}")
					
					if USE_EMAIL_NOTIFICATION:
						send_email(msg)
		prev_epoch_averages = epoch_averages

		if (CHECKPOINT_SAVE_RATE is not None) and (epoch % CHECKPOINT_SAVE_RATE == 0) and (epoch != 0):
			save_path = os.path.join(os.path.join(BASE_FILEPATH, CHECKPOINT_PATH), f"e{epoch}.ckpt")
			diffusion_model.save_weights(save_path)
			latest_save_path = save_path
			logger(f"""
								Reached checkpoint at Epoch {epoch}!
								\n\n\n
								Epoch Averages: {str(epoch_averages)}
								""")
			
			if USE_EMAIL_NOTIFICATION:
				send_email(f"""
								Reached checkpoint at Epoch {epoch}!
								\n\n\n
								Epoch Averages: {str(epoch_averages)}
								""")

		if (RESULT_SAMPLE_RATE is not None) and (epoch % RESULT_SAMPLE_RATE == 0) and (epoch != 0):
			prev_frames_sample_batch, _ = next(test_dataloader)
			print(tf.shape(prev_frames_sample_batch))
			prev_frames_sample_reshaped  =  tf.reshape(
				tf.transpose(
					prev_frames_sample_batch,
					[0, 2, 3, 1, 4]),
					(SAMPLE_BATCH_SIZE, ) + LATENT_SHAPE[:-1] + (-1, ))
			print(tf.shape(prev_frames_sample_reshaped))
			new_frames = diffusion_model.sample_from_frames(
				prev_frames_sample_reshaped,
				num_frames = 4,
				batch_size = SAMPLE_BATCH_SIZE).numpy()
			
			# Min-max scale and transform to uint8 for storage efficiency
			new_frames_min_shift = (new_frames - new_frames.min())
			new_frames_uint = ((new_frames_min_shift / new_frames_min_shift.max()) * 255.).astype("uint8")

			# Save sample to results directory
			save_path = os.path.join(os.path.join(BASE_FILEPATH, RESULT_DIRPATH), f"e{epoch}_sample.npy")
			with open(save_path, "wb") as npy_file:
				np.save(npy_file, new_frames_uint)

			# TODO: turn sample into video and send email
				
	if (CHECKPOINT_SAVE_RATE is None) or (epoch % CHECKPOINT_SAVE_RATE != 0):
		save_path = os.path.join(os.path.join(BASE_FILEPATH, CHECKPOINT_PATH), f"e{epoch}.ckpt")
		diffusion_model.save_weights(save_path)

	logger("Training Completed!")
		
	if USE_EMAIL_NOTIFICATION:
		send_email("Training Completed!")


if __name__ == "__main__":
	main()
	# try:
	# 	main()
	# except Exception as e:
	# 	if USE_EMAIL_NOTIFICATION:
	# 		send_email(f"Process exited with an error: {str(e)}")
	# 	else:
	# 		print(f"Process exited with an error: {str(e)}")