import sdkit
from sdkit.models import load_model
from sdkit.filter import apply_filters
from PIL import Image

context = sdkit.Context()
image = Image.open("photo of a man.jpg")

# set the path to the model file on the disk
context.model_paths["gfpgan"] = "C:\\path\\to\\gfpgan-1.3.pth"
load_model(context, "gfpgan")

# apply the filter
image_face_fixed = apply_filters(context, "gfpgan", image)

# save the filtered image
image_face_fixed.save("man_face_fixed.jpg")
