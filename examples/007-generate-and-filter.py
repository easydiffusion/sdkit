import sdkit
from sdkit.filter import apply_filters
from sdkit.generate import generate_images
from sdkit.models import load_model
from sdkit.utils import save_images

context = sdkit.Context()

# setup model paths
context.model_paths["stable-diffusion"] = "D:\\path\\to\\modelA.ckpt"
context.model_paths["gfpgan"] = "C:\\path\\to\\gfpgan-1.3.pth"
load_model(context, "stable-diffusion")
load_model(context, "gfpgan")

# generate image
images = generate_images(
    context,
    prompt="Photograph of an astronaut riding a horse",
    seed=42,
    width=512,
    height=512,
    hypernetwork_strength=0.3,
)

# apply filter
images_face_fixed = apply_filters(context, filters="gfpgan", images=images)

# save images
save_images(
    images_face_fixed,
    dir_path="D:\\path\\to\\images\\directory",
    file_name="image_with_face_fix",
)
