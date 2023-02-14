import sdkit
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import log, save_images

context = sdkit.Context()

# set the path to the model file on the disk (.ckpt or .safetensors file)
context.model_paths["stable-diffusion"] = "D:\\path\\to\\512-base-ema.ckpt"
load_model(context, "stable-diffusion")

# generate the image
images = generate_images(context, prompt="Photograph of an astronaut riding a horse", seed=42, width=512, height=512)

# save the image
save_images(images, dir_path=".")

log.info("Generated images!")
