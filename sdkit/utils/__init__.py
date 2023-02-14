import logging

log = logging.getLogger('sdkit')
LOG_FORMAT = '%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s %(message)s'
logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt="%X"
)

from .file_utils import (
    load_tensor_file,
    save_tensor_file,
    save_images,
    save_dicts,
)

from .hash_utils import (
    hash_bytes,
    hash_url_quick,
    hash_file_quick,
)

from .image_utils import (
    img_to_base64_str,
    img_to_buffer,
    buffer_to_base64_str,
    base64_str_to_buffer,
    base64_str_to_img,
    resize_img,
    apply_color_profile,
)

from .latent_utils import (
    img_to_tensor,
    get_image_latent_and_mask,
    latent_samples_to_images,
)

from .memory_utils import (
    gc,
    get_device_usage,
    print_largest_tensors_in_memory,
    print_tensor_info,
    get_object_id,
    record_tensor_name,
    get_tensors_in_memory,
    take_memory_snapshot,
)

from .http_utils import (
    download_file,
)