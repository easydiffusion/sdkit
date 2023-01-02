import sdkit
from sdkit.generate import get_cond_and_uncond
from sdkit.utils import log, floatArrayToGrayscaleBitmap, save_images

# Convert a prompt conditionings to a bitmap.

context = sdkit.Context()

model_path = 'D:\\path\\to\\malicious_model.ckpt'
load_model(context, 'stable-diffusion')

prompt = 'a photograph of an astronaut riding a horse'
unconditional_prompt = ' '
num_outputs = 1

conditioning, unconditional_conditioning = get_cond_and_uncond(prompt, unconditional_prompt, num_outputs, model)

images = floatArrayToGrayscaleBitmap(conditioning)
img_out_path = get_base_path(req.save_to_disk_path, req.session_id, req.prompt, 'cond', 'bmp')
print('floatArrayToGrayscaleBitmap', images)
save_images(images, img_out_path, 'bmp')

images = floatArrayToGrayscaleBitmap(unconditional_conditioning)
img_out_path = get_base_path(req.save_to_disk_path, req.session_id, req.prompt, 'uncond', 'bmp')
print('floatArrayToGrayscaleBitmap', images)
save_images(images, img_out_path, 'bmp')
save_images(images, dir_path: str, file_name='image', output_format='BMP'):
