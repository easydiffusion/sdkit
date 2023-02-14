import torch

from sdkit.utils import log

def get_cond_and_uncond(prompt, negative_prompt, batch_size, model):
    cond = parse_prompt(prompt, batch_size, model)
    uncond = parse_prompt(negative_prompt, batch_size, model)

    return cond, uncond

def parse_prompt(prompt, batch_size, model):
    """
    Requires model to be on the device
    """
    empty_result = model.get_learned_conditioning(batch_size * [""])
    result = torch.zeros_like(empty_result)
    subprompts, weights = split_weighted_subprompts(prompt)
    weights_sum = sum(weights)

    for i, subprompt in enumerate(subprompts):
        result = torch.add(result, model.get_learned_conditioning(batch_size * [subprompt]), alpha=weights[i] / weights_sum)

    if len(subprompts) == 0:
        result = empty_result

    return result

def split_weighted_subprompts(text):
    """
    grabs all text up to the first occurrence of ':' 
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        if ":" in text:
            idx = text.index(":") # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx+1:]
            # find value for weight 
            if " " in text:
                idx = text.index(" ") # first occurence
            else: # no space, read to end
                idx = len(text)
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except: # couldn't treat as float
                    log.warn(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                    weight = 1.0
            else: # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx+1:]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(weight)
        else: # no : found
            if len(text) > 0: # there is still text though
                # take remainder as weight 1
                prompts.append(text)
                weights.append(1.0)
            remaining = 0
    return prompts, weights