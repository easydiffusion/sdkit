import sdkit
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import log, save_images

context = sdkit.Context()

# set the path to the model and hypernetwork file on the disk
context.model_paths["stable-diffusion"] = "D:\\path\\to\\model.ckpt"
context.model_paths["hypernetwork"] = "D:\\path\\to\\hypernetwork.pt"
load_model(context, "stable-diffusion")
load_model(context, "hypernetwork")

# generate the image, hypernetwork_strength at 0.3
images = generate_images(
    context,
    prompt="Photograph of an astronaut riding a horse",
    seed=42,
    width=512,
    height=512,
    hypernetwork_strength=0.3,
)

# save the image
save_images(images, dir_path="D:\\path\\to\\images\\directory")

log.info("Generated images with a custom VAE!")
