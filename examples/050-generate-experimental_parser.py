import sdkit
from sdkit.models import load_model
from sdkit.generate import generate_images
from sdkit.utils import save_images, log

# experimental parser, which provides better control over the assignment of weights to tokens,
# as well as smoother transition between concepts

# the experimental parser currently requires the prompt to start with an exclamation mark: `!`
# e.g. `!this is my prompt`
# otherwise, sdkit will use the original parser.

# work-in-progress!

context = sdkit.Context()

# set the path to the model file on the disk (.ckpt or .safetensors file)
context.model_paths["stable-diffusion"] = "F:/models/stable-diffusion/sd-v1-4.ckpt"
load_model(context, "stable-diffusion")

# generate the image
for i, prompt in enumerate(
    [
        "!A photograph of an astronaut riding a horse",
        "!A [photograph] of an ((astronaut riding)) a (horse)",
        "!A [photograph] of an [[old] astronaut [[riding]]] a (horse)",
        '!A photograph of an (astronaut [riding]) a "horse:1 camel:1"',
        "!A ([photograph]) of an ([astronaut (riding)]) a [horse)",
        "!A photograph of an (astronaut:5 Monkey:3) riding a [horse:33 bike:55]",
        '!A photograph [[f/1.8]] of an (astronaut:5 "Monkey":3) riding a horse bike',
        "!(An old astronaut):20 [a cunning monkey]:80",
        '!(An old astronaut:80) "a cunning monkey":20',
        '!"(An old astronaut):80 (a cunning monkey):20"',
        '!A small "Astronaut:20 Monkey:40":10 Green:5',
        "!(Astronaut:50 Purple:5)",
        "!((Astronaut:2):50 Purple:5)",
        "!Astronaut:50 Purple:5",
        "!(Astronaut:50 Purple:5):50 Monkey:3 blue:1 green:2",
        "![Astronaut:0.75 Purple:-0.75]",
        "![Astronaut:-0.75 Purple:0.75]",
        "!((Astronaut:50 (Purple:5):50 (((Monkey:3 ([blue:1] [[green:2)",
    ]
):
    images = generate_images(context, prompt=prompt, seed=42, width=512, height=512)

    # save the image
    save_images(images, dir_path=".", file_name=f"prompt_{i}")

log.info("Generated images!")
