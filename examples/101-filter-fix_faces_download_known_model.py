import sdkit
from sdkit.models import download_model, resolve_downloaded_model_path, load_model
from sdkit.filter import apply_filters
from PIL import Image

context = sdkit.Context()
image = Image.open('photo of a man.jpg')

# download the model (skips if already downloaded, resumes if downloaded partially)
download_model(model_type='gfpgan', model_id='1.3')

# set the path to the auto-downloaded model
context.model_paths['gfpgan'] = resolve_downloaded_model_path(context, 'gfpgan', '1.3')
load_model(context, 'gfpgan')

# apply the filter
image_face_fixed = apply_filters(context, 'gfpgan', image)

# save the filtered image
image_face_fixed.save('man_face_fixed.jpg')
