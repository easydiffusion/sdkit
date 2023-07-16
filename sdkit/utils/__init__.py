import logging

log = logging.getLogger("sdkit")
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%X")


from .file_utils import load_tensor_file, save_dicts, save_images, save_tensor_file
from .hash_utils import hash_bytes, hash_file_quick, hash_url_quick
from .http_utils import download_file
from .image_utils import (
    apply_color_profile,
    base64_str_to_buffer,
    base64_str_to_img,
    buffer_to_base64_str,
    img_to_base64_str,
    img_to_buffer,
    resize_img,
)
from .latent_utils import (
    get_image_latent_and_mask,
    img_to_tensor,
    latent_samples_to_images,
    diffusers_latent_samples_to_images,
    to_tensor,
    tensor_to_bitmap,
)
from .memory_utils import (
    gc,
    get_device_usage,
    get_object_id,
    get_tensors_in_memory,
    print_largest_tensors_in_memory,
    print_tensor_info,
    record_tensor_name,
    take_memory_snapshot,
)
from .convert_model_utils import (
    convert_pipeline_unet_to_onnx,
    convert_pipeline_unet_to_tensorrt,
)
from .device_utils import has_amd_gpu
from .misc_utils import make_sd_context, get_nested_attr


def get_embedding_token(filename):
    return filename.split(".")[0].replace(" ", "_")
