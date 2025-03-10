import setuptools

setuptools.setup(
    install_requires=[
        "setuptools==69.5.1",
        "stable-diffusion-sdkit==2.1.5",  # wrapper around stable-diffusion, to allow pip install
        "gfpgan",
        "piexif",
        "realesrgan",
        "requests",
        "picklescan",
        "safetensors==0.3.3",
        "k-diffusion==0.0.12",
        "diffusers==0.28.2",
        "compel==2.0.1",
        "accelerate==0.23.0",
        "controlnet-aux==0.0.6",
        "invisible-watermark==0.2.0",  # required for SD XL
        "huggingface_hub==0.21.4",
        "torchruntime>=1.13.1",
        "numpy<2",
    ],
)
