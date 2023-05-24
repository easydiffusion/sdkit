import sdkit
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import log, save_images

context = sdkit.Context()
context.test_diffusers = True

# set the path to the model and VAE file on the disk
context.model_paths["lora"] = "D:\\path\\to\\lora.safetensors"
context.model_paths["stable-diffusion"] = "D:\\path\\to\\model.ckpt"
load_model(context, "stable-diffusion")
load_model(context, "lora")

# generate the image
images = generate_images(context, prompt="Photograph of an astronaut riding a horse", seed=42, width=512, height=512)

# save the image
save_images(images, dir_path="D:\\path\\to\\images\\directory")

log.info("Generated images with a custom VAE!")
