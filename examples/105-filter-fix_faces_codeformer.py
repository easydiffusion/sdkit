from PIL import Image

import sdkit
from sdkit.filter import apply_filters
from sdkit.models import load_model

context = sdkit.Context()
image = Image.open("photo of a man.png")

# set the path to the model file on the disk
context.model_paths["codeformer"] = "C:/path/to/codeformer.pth"
load_model(context, "codeformer")

# apply the filter
image_face_fixed = apply_filters(context, "codeformer", image)

# save the filtered image
image_face_fixed[0].save(f"man_face_fixed_codeformer.jpg")
