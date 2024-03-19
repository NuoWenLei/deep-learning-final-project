# deep-learning-final-project

[Project Drive](https://drive.google.com/drive/u/1/folders/10sVDeDwl1yXm8iJd95UBQhUD-DNItSTW)

We use the pretrained VAE Image Auto-encoders from Keras-CV.

- [Image Encoder](https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/image_encoder.py#L25)
- [Image Decoder](https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/stable_diffusion/decoder.py)

We re-implement the [Classic U-Net](https://arxiv.org/abs/1505.04597) introduced in 2015, however we train it in a pretrained Stable Diffusion Latent Space in order to reduce the model size. Then we use a method called "frame-stacking" as outlined in [Diffusion World Model](https://openreview.net/pdf?id=bAXmvOLtjA) to do frame-by-frame **unguided** video diffusion. The original frame-stacking model used a [classic U-Net](https://arxiv.org/abs/1505.04597) directly in the pixel space, which resulted in an architecture with 122 million parameters. We hope to reduce the number of trainable parameters by training in a pretrained latent space.
