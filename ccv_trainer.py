from imports import tf, np, tqdm

from unguided_diffusion.helpers import create_flow_unguided, load_latent_data
from unguided_diffusion.diffusion import UnguidedDiffusion

import gc
import json
import datetime
from functools import partial
from collections import defaultdict

from emailSender import send_attached_email, send_email
from constants import (
	# Training Settings
	LOG_FILEPATH,
	RESULT_DIRPATH,
	LATENT_SAMPLE_PATH,
	BATCH_SIZE,
	NUM_EPOCHS,
	USE_SAMPLE_DATA,
	
	# Model Params
	LATENT_SHAPE,
	NUM_FILTERS,
	NUM_BLOCKS,
	DROPOUT_PROB,
	FILTER_SIZE,
	NOISE_LEVELS,
	LEAKY,
	
	# Optimizer
	LEARNING_RATE)


def log(msg, filepath):
	with open(filepath, "a") as f:
		datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		f.write(f"\n{datetimeStr}: \n {msg}\n")
	return msg

def main():

	logger = partial(log, filepath=LOG_FILEPATH)

	if USE_SAMPLE_DATA:
		sample_latents = load_latent_data(LATENT_SAMPLE_PATH)
		NUM_SAMPLES = sample_latents.shape[0]
		dataloader = create_flow_unguided(sample_latents, batch_size = BATCH_SIZE)
		logger("USE_SAMPLE_DATA=True, using sample data")

	diffusion_model = UnguidedDiffusion(
		input_shape=LATENT_SHAPE[:2],
		num_channels=LATENT_SHAPE[-1],
		batch_size = BATCH_SIZE,
		start_filters=NUM_FILTERS,
		num_blocks=NUM_BLOCKS,
		dropout_prob=DROPOUT_PROB,
		filter_size=FILTER_SIZE,
		noise_level=NOISE_LEVELS,
		leaky=LEAKY,
		name="unguided_diffusion"
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
			batch = next(dataloader)
			metrics = diffusion_model.train_step(batch)
			pb.add(BATCH_SIZE, values=[(k, v) for k,v in metrics.items()])

		epoch_averages = dict((k, v[0] / v[1]) for k, v in pb._values.items())
		for k in epoch_averages.keys():
			summed_metrics[k] += epoch_averages[k]

		# TODO:
		# - checkpoints
		# - email updates
		# - callbacks







	# epoch = 0
	# step = 0
	
	# scaler = torch.cuda.amp.GradScaler()

	# checkpoint_path = args["resDirPath"] + "checkpoint-audio-diffusion.pt"

	# model.train()
	# while epoch < 100:
	# 	avg_loss = 0
	# 	avg_loss_step = 0
	# 	progress = tqdm(range(dataloader.numBatch))
	# 	for i in progress:
	# 		audio, caption = dataloader.nextBatch()
	# 		with torch.autograd.set_detect_anomaly(True):
	# 			optimizer.zero_grad()
	# 			audio = torch.from_numpy(audio).to(device)
	# 			with torch.cuda.amp.autocast():
	# 				loss = model(audio, text=caption.tolist())
	# 				avg_loss += loss.item()
	# 				avg_loss_step += 1
	# 			scaler.scale(loss).backward()
	# 			scaler.step(optimizer)
	# 			scaler.update()
	# 			progress.set_postfix(
	# 				loss=loss.item(),
	# 				epoch=epoch + i / dataloader.numBatch,
	# 			)

	# 			if (step % 1000 == 0) and (step != 0):
	# 				cap = caption.tolist()[-1]
	# 				send_email(logger(f"Step {step} Sample caption: {cap}"))
	# 				# Turn noise into new audio sample with diffusion
	# 				noise = torch.randn(1, 1, NUM_SAMPLES, device=device)
	# 				with torch.cuda.amp.autocast():
	# 					sample = model.sample(noise, text=[cap], num_steps=100)

	# 				save_path = args["resDirPath"] + f'test_generated_sound_{step}.wav'
	# 				torchaudio.save(save_path, sample[0].cpu(), SAMPLE_RATE)
	# 				del sample
	# 				gc.collect()
	# 				torch.cuda.empty_cache()
	# 				send_attached_email(f'CCV Step {step} Sample', save_path)
				
	# 			if step % 100 == 0:
	# 				msg = logger(f"Step {step}, Epoch {epoch + i / dataloader.numBatch}, loss {avg_loss / avg_loss_step}")
	# 				avg_loss = 0
	# 				avg_loss_step = 0
	# 				if step % 500 == 0:
	# 					send_email(msg)
				
	# 			del audio
	# 			del caption
	# 			gc.collect()

	# 			step += 1

	# 	epoch += 1
	# 	torch.save({
	# 		'epoch': epoch,
	# 		'model_state_dict': model.state_dict(),
	# 		'optimizer_state_dict': optimizer.state_dict(),
	# 	}, checkpoint_path)


# def parse_args():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument("--checkpoint", type=str, default=None)
# 	parser.add_argument("--resume", action="store_true")
# 	parser.add_argument("--run_id", type=str, default=None)
# 	return parser.parse_args()


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		send_email(f"Process Exited with an error: {str(e)}")