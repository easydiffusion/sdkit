from PIL import Image

import sdkit
from sdkit.filter import apply_filters
from sdkit.models import load_model

context = sdkit.Context()
image = Image.open("photo of a man.jpg")

# set the path to the model file on the disk
context.model_paths["realesrgan"] = "C:\\path\\to\\RealESRGAN_x4plus.pth"
load_model(context, "realesrgan")

# apply the filter
scale = 4  # or 2
image_upscaled = apply_filters(context, "realesrgan", image, scale=scale)

# save the filtered image
image_upscaled.save("man_upscaled.jpg")
