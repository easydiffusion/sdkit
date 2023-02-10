import sdkit
from sdkit.generate import generate_images
from sdkit.models import load_model

context = sdkit.Context()
context.model_paths["stable-diffusion"] = "D:\\path\\to\\sd-v1-4.ckpt"
context.device = "cpu"

load_model(context, "stable-diffusion")

# generate image
images = generate_images(
    context,
    prompt="Photograph of an astronaut riding a horse",
    seed=42,
    width=512,
    height=512,
)
images[0].save("image_from_cpu.jpg")
