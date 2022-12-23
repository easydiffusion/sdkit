import sdkit
from sdkit.models import load_model
from sdkit.generate import generate_images
from sdkit.utils import save_images, log

context = sdkit.Context()

# set the path to the model and VAE file on the disk
context.model_paths['stable-diffusion'] = 'D:\\path\\to\\model.ckpt'
context.model_paths['vae'] = 'D:\\path\\to\\vae.ckpt'
load_model(context, 'stable-diffusion')
load_model(context, 'vae')

# generate the image
images = generate_images(context, prompt="Photograph of an astronaut riding a horse", seed=42, width=512, height=512)

# save the image
save_images(images, dir_path='D:\\path\\to\\images\\directory')

log.info('Generated images with a custom VAE!')
