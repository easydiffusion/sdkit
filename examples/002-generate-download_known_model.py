import sdkit
from sdkit.generate import generate_images
from sdkit.models import download_model, load_model, resolve_downloaded_model_path
from sdkit.utils import log, save_images

context = sdkit.Context()

# download the model (skips if already downloaded, resumes if downloaded partially)
download_model(model_type='stable-diffusion', model_id='1.5-pruned-emaonly')

# set the path to the auto-downloaded model
context.model_paths['stable-diffusion'] = resolve_downloaded_model_path(context, 'stable-diffusion', '1.5-pruned-emaonly')
load_model(context, 'stable-diffusion')

# generate the image
images = generate_images(context, prompt='Photograph of an astronaut riding a horse', seed=42, width=512, height=512)

# save the image
save_images(images, dir_path='D:\\path\\to\\images\\directory')

log.info("Generated images!")
