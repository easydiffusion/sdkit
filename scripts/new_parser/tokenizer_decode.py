from pprint import pformat
import sdkit
from sdkit.models import load_model
from sdkit.generate import parse_prompt

# Parser test/Demo

context = sdkit.Context()

# setup model paths
context.model_paths["stable-diffusion"] = "D:\\path\\to\\modelA.ckpt"
load_model(context, "stable-diffusion")

for original_prompt in [
    "A photograph of an astronaut riding a horse",
    "A [photograph] of an ((astronaut riding)) a (horse)",
    "A [photograph] of an [[old] astronaut [[riding]]] a (horse)",
    'A photograph of an (astronaut [riding]) a "horse:1 camel:1"',
    "A ([photograph]) of an ([astronaut (riding)]) a [horse)",
    "A photograph of an (astronaut:5 Monkey:3) riding a [horse:33 bike:55]",
    'A photograph [[f/1.8]] of an (astronaut:5 "Monkey":3) riding a horse bike',
    "(An old astronaut):20 [a cunning monkey]:80",
    '(An old astronaut:80) "a cunning monkey":20',
    '"(An old astronaut):80 (a cunning monkey):20"',
    'A small "Astronaut:20 Monkey:40":10 Green:5',
    "(Astronaut:50 Purple:5)",
    "((Astronaut:2):50 Purple:5)",
    "Astronaut:50 Purple:5",
    "(Astronaut:50 Purple:5):50 Monkey:3 blue:1 green:2",
    "[Astronaut:0.75 Purple:-0.75]",
    "[Astronaut:-0.75 Purple:0.75]",
    "((Astronaut:50 (Purple:5):50 (((Monkey:3 ([blue:1] [[green:2)",
]:
    print("Prompt:", original_prompt)
    prompt, transforms = parse_prompt(original_prompt)
    print("  Parsed:", prompt)
    # print_text_conditioning(model, prompt)
    if transforms:
        print("  Transforms:")
        for t in transforms:
            print(pformat(t, indent=1, width=128, compact=False, sort_dicts=False))
    else:
        print("  Transforms: None")
