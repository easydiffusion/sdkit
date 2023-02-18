from sdkit import Context
from sdkit.models import load_model
from sdkit.filter import apply_filters

from PIL import Image

c = Context()

load_model(c, "nsfw_checker")

img = Image.open("image.jpg")

# the image will be blurred if NSFW content is detected.
# otherwise, the original image will be returned.
images = apply_filters(c, "nsfw_checker", img)

images[0].save("filtered.jpg")
