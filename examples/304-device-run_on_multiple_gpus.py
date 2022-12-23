import sdkit
from sdkit.models import load_model
from sdkit.generate import generate_images
from sdkit.utils import log

import threading

def render_thread(device):
    log.info(f'starting on device {device}')
    context = sdkit.Context()
    context.model_paths['stable-diffusion'] = 'D:\\path\\to\\sd-v1-4.ckpt'
    context.device = device

    load_model(context, 'stable-diffusion')

    # generate image
    log.info(f'generating on device {device}')
    images = generate_images(context, prompt='Photograph of an astronaut riding a horse', seed=42, width=512, height=512)
    images[0].save(f'image_from_{device}.jpg')

    log.info(f'finished generating on device {device}')

def start_thread(device):
    thread = threading.Thread(target=render_thread, kwargs={'device': 'cuda:0'})
    thread.daemon = True
    thread.name = f'SD-{device}'
    thread.start()

# assuming the PC has two CUDA-compatible GPUs, start on the first two GPUs: cuda:0 and cuda:1
start_thread('cuda:0')
start_thread('cuda:1')
