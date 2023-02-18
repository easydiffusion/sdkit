import setuptools

setuptools.setup(
    install_requires=[
        "stable-diffusion-sdkit==2.1.3",  # wrapper around stable-diffusion, to allow pip install
        "gfpgan",
        "piexif",
        "realesrgan",
        "requests",
        "picklescan",
        "safetensors",
        "k-diffusion",
        "diffusers",
    ],
)
