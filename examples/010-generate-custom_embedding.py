import sdkit
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import log, save_images

context = sdkit.Context()

# load the stable diffusion model
context.model_paths["stable-diffusion"] = "D:\\path\\to\\model.ckpt"
load_model(context, "stable-diffusion")

# load the embedding file
# the embedding token is the name of the file (without the file extension)
context.model_paths["embeddings"] = "D:\\path\\to\\demo-embedding.safetensors"  # this can also be a list of paths
load_model(context, "embeddings")

# generate the image
images = generate_images(
    context,
    prompt="demo-embedding Photograph of an astronaut riding a horse",
    seed=42,
    width=512,
    height=512,
    lora_alpha=0.3,
)

# save the image
save_images(images, dir_path="D:\\path\\to\\images\\directory")

log.info("Generated images with a custom embedding!")
