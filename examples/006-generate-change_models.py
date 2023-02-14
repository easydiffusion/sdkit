# first we'll generate an image using modelA.ckpt and a hypernetwork.
# then we'll generate another image using modelB.ckpt, without a hypernetwork.
# the unused hypernetwork and modelA.ckpt will be unloaded from memory automatically.

import sdkit
from sdkit.models import load_model, unload_model
from sdkit.generate import generate_images
from sdkit.utils import save_images

context = sdkit.Context()

# first image with modelA.ckpt, with hypernetwork
context.model_paths['stable-diffusion'] = 'D:\\path\\to\\modelA.ckpt'
context.model_paths['hypernetwork'] = 'D:\\path\\to\\hypernetwork.pt'
load_model(context, 'stable-diffusion')
load_model(context, 'hypernetwork')

images = generate_images(context, prompt="Photograph of an astronaut riding a horse", seed=42, width=512, height=512, hypernetwork_strength=0.3)

save_images(images, dir_path='D:\\path\\to\\images\\directory', file_name='image_modelA_with_hypernetwork')

# second image with modelB.ckpt, without hypernetwork
context.model_paths['stable-diffusion'] = 'D:\\path\\to\\modelB.ckpt'
context.model_paths['hypernetwork'] = None
load_model(context, 'stable-diffusion')
unload_model(context, 'hypernetwork')

images = generate_images(context, prompt="Photograph of an astronaut riding a horse", seed=42, width=512, height=512)

save_images(images, dir_path='D:\\path\\to\\images\\directory', file_name='image_modelB_without_hypernetwork')
