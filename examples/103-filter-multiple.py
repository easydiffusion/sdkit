from PIL import Image

import sdkit
from sdkit.filter import apply_filters
from sdkit.models import load_model

context = sdkit.Context()
image = Image.open("photo of a man.jpg")

# set the path to the model files on the disk
context.model_paths = {
    "gfpgan": "C:\\path\\to\\gfpgan-1.3.pth",
    "realesrgan": "C:\\path\\to\\realesrgan.pth",
}
load_model(context, "gfpgan")
load_model(context, "realesrgan")

# apply the filters
image_face_fixed = apply_filters(context, "gfpgan", image)
image_scaled_up = apply_filters(context, "realesrgan", image)
image_face_fixed_and_scaled_up = apply_filters(context, ["gfpgan", "realesrgan"], image)

# save the filtered images
image_face_fixed.save("man_face_fixed.jpg")
image_scaled_up.save("man_scaled_up.jpg")
image_face_fixed_and_scaled_up.save("man_face_fixed_and_scaled_up.jpg")
