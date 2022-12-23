This is a community project, so please feel free to contribute (and use in your project)!

[![Discord Server](https://img.shields.io/discord/1014774730907209781?label=Discord)](https://discord.com/invite/u9yhsFmEkB)

![t2i](https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/assets/stable-samples/txt2img/768/merged-0006.png)

# Contributing
We'd love to accept code contributions. Please feel free to drop by our [Discord community](https://discord.com/invite/u9yhsFmEkB)!

📢 We're looking for code contributions for these features (or anything else you'd like to work on):
- CodeFormer upscaling (please maintain the required copyright notices).
- Using custom Textual Inversion embeddings.
- Seamless tiling.
- Outpainting.
- Mac support.
- AMD support.
- Allow other samplers for img2img (instead of only DDIM).

# Setting up a developer environment
If you're on Windows or Linux, please install CUDA-compatible `torch` and `torchvision` (if you haven't already) using: `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116`

1. Create a fork of this repository using https://github.com/easydiffusion/sdkit/fork
2. Copy the link to your repository using the "Code" button in your repository. Important: Use YOUR fork for this.
![image](https://user-images.githubusercontent.com/844287/209371553-38ef7144-897e-4211-a186-5a235ff71375.png)

3. Install an editable copy of your fork using: `pip install -e git+git@github.com:YOUR_FORK_NAME/sdkit.git#egg=sdkit` or `pip install -e git+https://github.com/YOUR_FORK_NAME/sdkit.git#egg=sdkit`. Replace `YOUR_FORK_NAME` with what your fork has.
4. This will checkout a working copy of sdkit, along with any required dependencies.
5. You can now edit the code and submit pull-requests for any changes you'd like to contribute.
