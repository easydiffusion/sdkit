import setuptools

setuptools.setup(
    install_requires=[
        "stable-diffusion-sdkit==2.1.4",  # wrapper around stable-diffusion, to allow pip install
        "gfpgan",
        "piexif",
        "realesrgan",
        "requests",
        "picklescan",
        "safetensors==0.3.2",
        "k-diffusion",
        "diffusers==0.19.2",
        "compel==2.0.1",
        "accelerate==0.18.0",
        "controlnet-aux==0.0.6",
        "invisible-watermark==0.2.0",  # required for SD XL
    ],
)
