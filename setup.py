import setuptools

setuptools.setup(
    install_requires=[
        "stable-diffusion-sdkit", # wrapper around stable-diffusion, to allow pip install

        "gfpgan",
        "realesrgan",
        "requests",
        "picklescan",
        "safetensors",
        "k-diffusion",
    ],
)
