import setuptools

setuptools.setup(
    install_requires=[
        "stable-diffusion-sdkit==2.1.4",  # wrapper around stable-diffusion, to allow pip install
        "gfpgan",
        "piexif",
        "realesrgan",
        "requests",
        "picklescan",
        "safetensors",
        "k-diffusion",
        "diffusers==0.14.0",
        "compel",
    ],
)
