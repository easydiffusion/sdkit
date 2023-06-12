import cv2
import torch

from sdkit import Context
from sdkit.models import load_model, unload_model

from torchvision.transforms.functional import normalize
from threading import Lock
from PIL import Image
import numpy as np

from basicsr.utils import img2tensor, tensor2img
from .face_restoration_helper import FaceRestoreHelper

codeformer_temp_device_lock = Lock()  # workaround: codeformer currently can only start on one device at a time.


def inference(context: Context, image, upscale_bg, upscale_faces, upscale_factor, codeformer_fidelity, codeformer_net):
    device = torch.device(context.device)
    face_helper = FaceRestoreHelper(upscale_factor=upscale_factor, use_parse=True, device=device)
    face_helper.clean_all()
    face_helper.read_image(image)
    face_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
    face_helper.align_warp_face()

    bg_upscaler = context.models["realesrgan"] if upscale_bg else None
    face_upscaler = context.models["realesrgan"] if upscale_faces else None

    # face restoration for each cropped face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = codeformer_net(cropped_face_t, w=codeformer_fidelity, adain=True)[0]
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except RuntimeError as error:
            print(f"Failed inference for CodeFormer: {error}")
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face)

    # paste_back
    face_helper.get_inverse_affine(None)
    bg_img = bg_upscaler.enhance(image, outscale=upscale_factor)[0] if bg_upscaler else None
    restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, face_upsampler=face_upscaler)
    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)

    return restored_img


def apply(
    context: Context,
    input_img,
    upscale_background=False,
    upscale_faces=False,
    upscale_factor=1,
    codeformer_fidelity=0.5,
    **kwargs,
):
    if not context.enable_codeformer:
        raise Exception(
            "Please set `context.enable_codeformer` to True, to use CodeFormer. By enabling CodeFormer, "
            + "you agree to comply by the CodeFormer license (including non-commercial use of CodeFormer): "
            + "https://github.com/sczhou/CodeFormer/blob/master/LICENSE"
        )
    if (upscale_background or upscale_faces) and "realesrgan" not in context.models:
        raise Exception("realesrgan not loaded in context.models! Required for upscaling in CodeFormer.")

    device = torch.device(context.device)
    codeformer_net = context.models["codeformer"]

    # Convert PIL Image to numpy array and ensure it's in BGR format for OpenCV
    input_img = np.array(input_img)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    # Run inference
    with codeformer_temp_device_lock:  # Wait for any other devices to complete before starting.
        # hack for a bug in facexlib: https://github.com/xinntao/facexlib/pull/19/files
        from facexlib.detection import retinaface

        retinaface.device = torch.device(context.device)

        result = inference(
            context, input_img, upscale_background, upscale_faces, upscale_factor, codeformer_fidelity, codeformer_net
        )

    pil_image = Image.fromarray(result)

    # Convert result back to RGB for PIL, then create PIL Image
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) # uncommenting this line turns people blue, which is a very convenient way to check CodeFormer is being used

    # Only resize if rescaling_factor is not 1
    if upscale_factor != 1:
        # Get original image dimensions
        original_width, original_height = pil_image.size

        # Calculate new dimensions
        new_width = int(original_width / upscale_factor)
        new_height = int(original_height / upscale_factor)

        # Resize the image, using the high-quality downsampling filter
        pil_image = pil_image.resize((new_width, new_height), Image.ANTIALIAS)

    # Return the rescaled/unchanged image
    return pil_image
