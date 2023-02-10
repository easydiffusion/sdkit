import sdkit
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import log, save_images

context = sdkit.Context()

# set the path to the custom model file on the disk
context.model_paths[
    "stable-diffusion"
] = "D:\\path\\to\\cmodelUpgradeStableD.safetensors"
context.model_configs["stable-diffusion"] = "D:\\path\\to\\Cmodelsafetensor.yaml"
# the yaml config file is required if it's an unknown model to use.
# it is not necessary for known models present in the models_db.

load_model(context, "stable-diffusion")

# generate the image
images = generate_images(
    context,
    prompt="Photograph of an astronaut riding a horse",
    seed=42,
    width=512,
    height=512,
)

# save the image
save_images(images, dir_path="D:\\path\\to\\images\\directory")

log.info("Generated images!")
