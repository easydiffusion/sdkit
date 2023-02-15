import sdkit
from sdkit.generate import get_cond_and_uncond
from sdkit.models import load_model
from sdkit.utils import log, tensor_to_bitmap, save_images

# Convert a prompt conditionings to a bitmap.

context = sdkit.Context()

context.model_paths["stable-diffusion"] = "D:\\path\\to\\model.ckpt"
load_model(context, "stable-diffusion")

prompt = "a photograph of an astronaut riding a horse"
unconditional_prompt = " "
num_outputs = 1

model = context.models["stable-diffusion"]

conditioning, unconditional_conditioning = get_cond_and_uncond(prompt, unconditional_prompt, num_outputs, model)

images = tensor_to_bitmap(conditioning)
log.info(f"tensor_to_bitmap: {images}")
save_images(images, dir_path=".", file_name="cond", output_format="BMP")

images = tensor_to_bitmap(unconditional_conditioning)
log.info(f"tensor_to_bitmap: {images}")
save_images(images, dir_path=".", file_name="uncond", output_format="BMP")
