import sdkit
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import log

from PIL import Image

context = sdkit.Context()


# convert an existing image into an openpose image (or skip these lines if you have a custom openpose image)
from sdkit.filter import apply_filters

input_image = Image.open("man_pose.jpg")  # set your input image here <---------

controlnet_filter_name = 'openpose'  # you can set this to other controlnet filters too, e.g. "canny" etc.
context.model_paths[controlnet_filter_name] = controlnet_filter_name
load_model(context, controlnet_filter_name)

filtered_image = apply_filters(context, controlnet_filter_name, input_image)

# load SD
context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"  # <---- SD model path here
load_model(context, "stable-diffusion")

# load controlnet
context.model_paths["controlnet"] = "models/controlnet/control_v11p_sd15_openpose.pth"  # <----- Controlnet model path
load_model(context, "controlnet")

# generate the image
image = generate_images(
    context,
    "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
    seed=42,
    width=512,
    height=768,
    control_image=filtered_image,
)[0]

# save the image
image.save("controlnet_image.jpg")

log.info("Generated images with a Controlnet!")
