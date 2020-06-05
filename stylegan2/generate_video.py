import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import tensorflow as tf
from progressbar import progressbar
import imageio
import random

def main(): 

	#Initialize TensorFlow
  tflib.init_tf()


# Load pre-trained network. 
  # model_path = "./model/gan_gen.pkl"

# Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
  global Gs
  model_path = "../trained_model/Gs_1024"
  with open(model_path,"rb") as f:
    Gs = pickle.load(f)

  fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

    # Randomize initial number to get a different image every time
  rnd = np.random.RandomState(random.randint(1,1000000))

  LATENT_SPACE_SCALE = 10.
  latent_vector = rnd.randn(1, Gs.input_shape[1]) * LATENT_SPACE_SCALE

  #Draw one image as starting point
  images = Gs.run(latent_vector, None, truncation_psi=1, randomize_noise=False, output_transform=fmt)
  PIL.Image.fromarray(images[0])

  def linear_morph(old_latent_vector, Gs, writer):

    new_latent_vector = rnd.randn(1, old_latent_vector.size) * LATENT_SPACE_SCALE

    for k in progressbar(np.arange(0., 1., 0.005)):
      latent_vector = old_latent_vector * (1. - k) + new_latent_vector * k
      images = Gs.run(latent_vector, None, truncation_psi=1, 
                      randomize_noise=False, output_transform=fmt)
      PIL.Image.fromarray(images[0])
      writer.append_data(images[0])

    return new_latent_vector


  def piecewise_linear_morph(latent_vector, Gs, writer, num_morphs):
    for i in range(num_morphs):
      latent_vector = linear_morph(latent_vector, Gs, writer)

  videopath = 'morphing_paintings_4.mp4'
  writer = imageio.get_writer(videopath, fps=30)

  piecewise_linear_morph(latent_vector, Gs, writer, num_morphs=20)

  writer.close()


if __name__ == "__main__":
  main()
