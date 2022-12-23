import sdkit
from sdkit.models import load_model
from sdkit.generate import generate_images

context = sdkit.Context()
context.model_paths['stable-diffusion'] = 'D:\\path\\to\\sd-v1-4.ckpt'
load_model(context, 'stable-diffusion')

# default (balanced) VRAM optimizations (much lower VRAM usage, performance is nearly as fast as max)
images = generate_images(context, prompt='Photograph of an astronaut riding a horse', seed=42, width=512, height=512)
images[0].save('image1.jpg')


# no VRAM optimizations (maximum VRAM usage, fastest performance)
context.vram_optimizations = {}
load_model(context, 'stable-diffusion') # reload the model, to apply the change to VRAM optimization

images = generate_images(context, prompt='Photograph of an astronaut riding a horse', seed=42, width=512, height=512)
images[0].save('image2.jpg')


# lowest VRAM usage, slowest performance (for GPUs with less than 4gb of VRAM)
context.vram_optimizations = {'KEEP_ENTIRE_MODEL_IN_CPU'}
load_model(context, 'stable-diffusion') # reload the model, to apply the change to VRAM optimization

images = generate_images(context, prompt='Photograph of an astronaut riding a horse', seed=42, width=512, height=512)
images[0].save('image3.jpg')
