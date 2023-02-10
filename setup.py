"""
    The setup file.
"""
import os
import setuptools

install_requires = []


if not os.getenv('SKIP_EXTERNAL_DEPS'):
    install_requires: list[str] = [
        # wrapper around stable-diffusion, to allow pip install
        "stable-diffusion-sdkit",

        "gfpgan",
        "realesrgan",
        "requests",
        "picklescan",
        "safetensors",
        "k-diffusion",
    ]

setuptools.setup(
    install_requires=install_requires,
)
