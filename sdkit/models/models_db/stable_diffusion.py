models = {
    # 2.1
    '2.1-512-ema-pruned': {
        'name': 'SD 2.1 Base 512x512 EMA',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt',
        'config_url': 'https://gist.githubusercontent.com/cmdr2/c12ab7c8a25b8239429866d7c68942ec/raw/27d92cae6301fad780421ed7b17a07e371b15b07/v2.1-inference.yaml',
        'quick_hash': '47c8ec7d3b1fc82c6f1f3ba20090b27e1885f83133f70762202619b22ef1e73b',
    },
    '2.1-512-nonema-pruned': {
        'name': 'SD 2.1 512x512 NonEMA',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-nonema-pruned.ckpt',
        'config_url': 'https://gist.githubusercontent.com/cmdr2/c12ab7c8a25b8239429866d7c68942ec/raw/27d92cae6301fad780421ed7b17a07e371b15b07/v2.1-inference.yaml',
        'quick_hash': 'd16726086cd9df3fd3c7fa701702b42bdc42f1f9a197aa7d3c25c42271952284',
    },
    '2.1-768-ema-pruned': {
        'name': 'SD 2.1 768x768 Pruned EMA',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt',
        'config_url': 'https://gist.githubusercontent.com/cmdr2/c12ab7c8a25b8239429866d7c68942ec/raw/27d92cae6301fad780421ed7b17a07e371b15b07/v2.1-inference.yaml',
        'quick_hash': '4bdfc29ccf12c8ed78fb249e0320e489595e422661d76f789e8145a196375e5d',
    },
    '2.1-768-nonema-pruned': {
        'name': 'SD 2.1 768x768 Pruned NonEMA',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-nonema-pruned.ckpt',
        'config_url': 'https://gist.githubusercontent.com/cmdr2/c12ab7c8a25b8239429866d7c68942ec/raw/27d92cae6301fad780421ed7b17a07e371b15b07/v2.1-inference.yaml',
        'quick_hash': 'e1542d5aab4c574a8abb85846408ba515d6310fffce529b04c8cdbcfcc644c0c',
    },

    # 2.0
    '2.0-512-base-ema': {
        'name': 'SD 2.0 Base 512x512',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt',
        'config_url': 'https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference.yaml',
        'quick_hash': '09dd2ae42846c479864af86d03d28617de028c0f0d55e5594de8d167e35b2cac',
    },
    '2.0-768-v-ema': {
        'name': 'SD 2.0 768x768 EMA v-prediction',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt',
        'config_url': 'https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml',
        'quick_hash': '2c02b20ada93991d3c6f2442fe50b740e982047af52de72e45ec718787b9c29d',
    },
    '2.0-512-depth-ema': {
        'name': 'SD 2.0 Depth 512x512 EMA',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2-depth/resolve/main/512-depth-ema.ckpt',
        'config_url': 'https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-midas-inference.yaml',
        'quick_hash': 'd0522d122e2713d38293c85e0b7b417e69234706fed2465ded3af0ac32d7e578',
    },
    '2.0-512-inpainting-ema': {
        'name': 'SD 2.0 Inpainting 512x512 EMA',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.ckpt',
        'config_url': 'https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inpainting-inference.yaml',
        'quick_hash': 'a13858304e838991400ffa6cc9d070e4e0577d74996e15c2a726eed89e1901b9',
    },
    '2.0-x4-upscaler-ema': {
        'name': 'SD 2.0 x4 Upscaling 512x512 EMA',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/resolve/main/x4-upscaler-ema.ckpt',
        'config_url': 'https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/x4-upscaling.yaml',
        'quick_hash': '5c5661de6783e42b25e013ad1fad13cb4f3cb84997f4334d6a8a5c3ad84a0e77',
    },

    # 1.5
    '1.5-pruned-emaonly': {
        'name': 'SD 1.5 Pruned EMA',
        'url': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt',
        'config_url': 'https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml',
        'quick_hash': '817611515b5da3b3092d709e3c777a861ece2ac161340bb6a8e99de04dae371d',
    },
    '1.5-pruned': {
        'name': 'SD 1.5 Pruned',
        'url': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt',
        'config_url': 'https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml',
        'quick_hash': 'a9263745ade4bdd9d1567c2e5e708f1d452314947711de7722cec6ee6396d81f',
    },
    '1.5-inpainting': {
        'name': 'SD 1.5 Pruned',
        'url': 'https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt',
        'config_url': 'https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inpainting-inference.yaml',
        'quick_hash': '3e16efc8bd8157a6d571660eb550fc73c318a134bf3c8718d2de66574b0cc004',
    },

    # 1.4
    '1.4': {
        'name': 'SD 1.4',
        'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt',
        'config_url': 'https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml',
        'quick_hash': '7460a6fa109d761365bc61ee2c22e8224215ce3182f1ed164799bd1dbd4d4cf9',
    },
    '1.4-full-ema': {
        'name': 'SD 1.4 EMA',
        'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt',
        'config_url': 'https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml',
        'quick_hash': '06c5042459bd99b7fd03156ea7b74519b6597b1716ba465882a4f04c39c8c337',
    },
}
