import os
import cv2
import torch
import torch.nn.functional as F

from sdkit import Context

from torchvision.transforms.functional import normalize

from PIL import Image
import numpy as np

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from sdkit.modules.codeformer.utils.misc import gpu_is_available, get_device
from sdkit.modules.codeformer.utils.realesrgan_utils import RealESRGANer
from sdkit.modules.codeformer.utils.registry import ARCH_REGISTRY
from sdkit.modules.codeformer.utils.logger import get_root_logger

from sdkit.modules.codeformer.archs.codeformer_arch import CodeFormer
from sdkit.modules.codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from sdkit.modules.codeformer.facelib.utils.misc import is_gray

from sdkit.models.model_loader.codeformer import initialize_models


"""
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
"""


def inference(image, background_enhance, face_upsample, upscale, codeformer_fidelity, device, codeformer_net, upsampler):
    """Run a single prediction on the model"""
    try: 
        # take the default setting for the demo
        has_aligned = False
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"

        img = image

        upscale = int(upscale) # convert type to int
        if upscale > 4: # avoid memory exceeded due to too large upscale
            upscale = 4 
        if upscale > 2 and max(img.shape[:2])>1000: # avoid memory exceeded due to too large img resolution
            upscale = 2 
        if max(img.shape[:2]) > 1500: # avoid memory exceeded due to too large img resolution
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img, upscale
    except Exception as error:
        print('Global exception', error)
        return None


def apply(context: Context, input_img, background_enhance=False, face_upsample=True, rescaling_factor=4, codeformer_fidelity=0, **kwargs):
    # Extract the model directory path for CodeFormer
    codeformer_path = context.model_paths['codeformer']
    model_dir = os.path.dirname(os.path.dirname(codeformer_path))
    upsampler, device, codeformer_net = initialize_models(context)

    # Convert PIL Image to numpy array and ensure it's in BGR format for OpenCV
    input_img = np.array(input_img)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    # Run inference
    result, rescaling_factor = inference(
            input_img,
            background_enhance,
            face_upsample,
            rescaling_factor,
            codeformer_fidelity,
            device,
            codeformer_net,
            upsampler
        )

    if result is not None:
        # Convert result back to RGB for PIL, then create PIL Image
        #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) # uncommenting this line turns people blue, which is a very convenient way to check CodeFormer is being used
        pil_image = Image.fromarray(result)

        # Get original image dimensions
        original_width, original_height = pil_image.size

        # Calculate new dimensions
        new_width = int(original_width / rescaling_factor)
        new_height = int(original_height / rescaling_factor)

        # Resize the image, using the high-quality downsampling filter
        rescaled_image = pil_image.resize((new_width, new_height), Image.ANTIALIAS)

        # Return the rescaled image
        return rescaled_image
    else:
        print('Error during inference.')
        return None
