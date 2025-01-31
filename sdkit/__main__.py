import argparse
import os
from PIL import Image

SUPPORTED_DEVICES = ("cpu", "mps", "cuda", "xpu", "directml", "mtia")


def main():
    parser = argparse.ArgumentParser(description="Generate images using sdkit.")
    parser.add_argument("-m", "--model", required=True, help="Path to the Stable Diffusion model.")
    parser.add_argument(
        "--type",
        choices=["f32", "f16"],
        default="f16",
        help="Weight type for precision (default: f16).",
    )
    parser.add_argument("-i", "--init-img", help="Path to the input image, required for img2img.")
    parser.add_argument(
        "-o",
        "--output",
        default="output.png",
        help="Path to save the result image (default: output.png).",
    )
    parser.add_argument("-p", "--prompt", required=True, help="The prompt to render.")
    parser.add_argument(
        "-n",
        "--negative-prompt",
        default="",
        help='The negative prompt (default: "").',
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=7.0,
        help="Unconditional guidance scale (default: 7.0).",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="Prompt strength for noising/unnoising (default: 0.75).",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=512,
        help="Image height in pixels (default: 512).",
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=512,
        help="Image width in pixels (default: 512).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of sampling steps (default: 20).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42; use random seed if < 0).",
    )
    parser.add_argument(
        "-b",
        "--batch-count",
        type=int,
        default=1,
        help="Number of images to generate (default: 1).",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device to use for rendering. Options: cpu, mps, cuda:0, directml:0, xpu:1 etc (default: automatic pick).",
    )

    args = parser.parse_args()

    import sdkit
    from sdkit.models import load_model
    from sdkit.generate import generate_images

    context = sdkit.Context()
    context.half_precision = args.type == "f16"

    if args.device:
        if not args.device.startswith(SUPPORTED_DEVICES):
            raise RuntimeError(f"Unsupported device: {args.device}! Supported devices: {SUPPORTED_DEVICES}")

        context.device = args.device

    context.model_paths["stable-diffusion"] = args.model
    load_model(context, "stable-diffusion")

    if args.init_img:
        if not os.path.exists(args.init_img):
            raise FileNotFoundError(f"Input image not found: {args.init_img}")
        init_image = Image.open(args.init_img).convert("RGB")
    else:
        init_image = None

    images = generate_images(
        context,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        init_image=init_image,
        seed=args.seed,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg_scale,
        prompt_strength=args.strength,
        num_outputs=args.batch_count,
    )

    o, ext = os.path.splitext(args.output)
    for i, img in enumerate(images):
        img.save(f"{o}_{i}{ext}")


if __name__ == "__main__":
    main()
