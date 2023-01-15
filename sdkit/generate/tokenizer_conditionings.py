"""tokenizer_conditionings.py: Prompt tokenizer and conditionings transforms.
Notes:
    conditioning: A tensor of 77 by (768 or 1024) float values.
        Each vector is a the representation of a token encoded into 768 dimensions for SD1 and 1024 for SD2.
        https://jalammar.github.io/illustrated-stable-diffusion/

    unconditional_conditioning: An input tensor mapping into the unguided latent space.
        The unconditional conditionings use

    unconditional_guidance_scale:
        https://benanne.github.io/2022/05/26/guidance.html

    SLERP:
        https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/

# Others
    https://github.com/JoaoLages/diffusers-interpret/tree/main/src/diffusers_interpret
    https://github.com/isaac-bender/stable_diffusion_interp/blob/main/DrEyeBender's_Stable_Diffusion_notebook_Public_copy.ipynb
"""
import torch
import numpy as np
import torch.nn.functional as F
from sdkit.utils import log, to_tensor
from .prompt_parser import clean_text, parse_prompt

# Default size for both SD1 and SD2 with open_CLIP
DEFAULT_VOCABULARY_SIZE = 49408
DEFAULT_TOKENS_LENGTH = 77

def decode_ids(encoder_model, ids):
    if hasattr(encoder_model, 'tokenizer') and encoder_model.tokenizer:
        tokenizer = encoder_model.tokenizer
        log.debug('Decoding using %s model', tokenizer.__class__.__name__)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        return ''.join([token.replace('</w>', ' ') for token in tokens])
    elif encoder_model.__class__.__name__ == 'FrozenOpenCLIPEmbedder':
        log.debug('Decoding using CLIPEmbedder model %s', encoder_model.__class__.__name__)
        import open_clip
        tokenizer = open_clip.tokenizer._tokenizer
        if not tokenizer:
            raise ReferenceError('open_clip.tokenizer._tokenizer is not defined.')
        return tokenizer.decode(ids)
    raise NotImplementedError(f'{encoder_model.__class__.__name__} has no tokenizer and is not implemented.')

def encode_tokens(encoder_model, text, wrapped=False, size=None):
    text = clean_text(text)
    if hasattr(encoder_model, 'tokenizer') and encoder_model.tokenizer:
        tokenizer = encoder_model.tokenizer
        log.debug('Encoding using %s model', tokenizer.__class__.__name__)
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        if size is not None:
            # Keep size minus the [Begining Of Sequence] and [End Of Sequence] tokens when wrapping.
            if wrapped:
                size = size - 2
            print_token_loss(encoder_model, tokens, max_size=size)
            tokens = tokens[0:size]
            ids = ids[0:size]
            if len(ids) < size:
                # pad up to size using Pad token or End Of Sequence tokens.
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id:
                    pad_token = tokenizer.pad_token
                    pad_token_id = tokenizer.pad_token_id
                else:
                    pad_token = tokenizer.eos_token
                    pad_token_id = tokenizer.eos_token_id
                tokens = tokens + [pad_token] * (size - len(ids))
                ids = ids + [pad_token_id] * (size - len(ids))
        if wrapped:
            tokens = [tokenizer.bos_token] + tokens + [tokenizer.eos_token]
            ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
        return tokens, ids
    log.debug('Loading embedder for model %s', encoder_model.__class__.__name__)
    if encoder_model.__class__.__name__ == 'FrozenOpenCLIPEmbedder':
        import open_clip
        tokenizer = open_clip.tokenizer._tokenizer
        if not tokenizer:
            raise ReferenceError('open_clip.tokenizer._tokenizer is not defined.')
        log.debug('Encoding using embedder %s', tokenizer.__class__.__name__)
        tokens = []
        text = open_clip.tokenizer.whitespace_clean(open_clip.tokenizer.basic_clean(text)).lower()
        import regex as re
        for token in re.findall(tokenizer.pat, text):
            token = ''.join(tokenizer.byte_encoder[b] for b in token.encode('utf-8'))
            tokens.extend(tokenizer.bpe(token).split(' '))
        ids = [tokenizer.encoder[token] for token in tokens]
        if size is not None:
            print_token_loss(encoder_model, tokens, max_size=size - 2 if wrapped else size)
        # open_clip doesn't seems to implement padding.
        # The tensor will be padded with zeroes.
        if wrapped:
            tokens = ["<start_of_text>"] + tokens + ["<end_of_text>"]
            sot_token = tokenizer.encoder["<start_of_text>"]
            eot_token = tokenizer.encoder["<end_of_text>"]
            ids = [sot_token] + ids + [eot_token]
        if size is not None and len(ids) > size:
            tokens = tokens[0:size]
            ids = ids[0:size]
            if wrapped:
                tokens[-1] = ["<end_of_text>"]
                ids[-1] = eot_token
        return tokens, ids
    raise NotImplementedError(f'{encoder_model.__class__.__name__} has no tokenizer and is not implemented.')

def get_token_length(encoder_model):
    if hasattr(encoder_model, 'max_length'):
        return encoder_model.max_length
    else:
        # The tokenizer default buffer length is 77
        # After without BOS and EOS there is 75 user tokens.
        log.warn(f'encoder_model.max_length is missing. Using {DEFAULT_TOKENS_LENGTH} as the default size.')
        return DEFAULT_TOKENS_LENGTH

def get_vocabulary_size(encoder_model):
    tokenizer = None
    if hasattr(encoder_model, 'tokenizer') and encoder_model.tokenizer:
        tokenizer = encoder_model.tokenizer
    elif encoder_model.__class__.__name__ == 'FrozenOpenCLIPEmbedder':
        import open_clip
        tokenizer = open_clip.tokenizer._tokenizer
        if not tokenizer:
            log.warn('open_clip.tokenizer._tokenizer is not defined.')
    if tokenizer:
        if hasattr(tokenizer, 'vocab_size'):
            return tokenizer.vocab_size
        if hasattr(tokenizer, 'encoder'):
            return len(tokenizer.encoder)
    log.warn(f'{encoder_model.__class__.__name__} has no vocab_size/encoder and/or is not implemented. Returning {DEFAULT_VOCABULARY_SIZE} as default.')
    return DEFAULT_VOCABULARY_SIZE

def print_token_loss(encoder_model, tokens, max_size=None):
    if max_size is None:
        max_size = get_token_length(encoder_model)
    if hasattr(encoder_model, 'tokenizer') and encoder_model.tokenizer:
        # when using CLIP tokenizers look for possible tokenizer.unk_token (unknown tokens)
        tokenizer = encoder_model.tokenizer
        if hasattr(tokenizer, 'unk_token'):
            if tokenizer.unk_token != (tokenizer.pad_token if hasattr(tokenizer, 'pad_token') else tokenizer.eos_token):
                unk_idx = [idx for idx, value in enumerate(tokens) if value == tokenizer.unk_token]
            else:
                unvalidated_unknowns = []
                unk_idx = []
                for idx, value in enumerate(tokens):
                    if value == tokenizer.unk_token:
                        unvalidated_unknowns.append(idx)
                    elif unvalidated_unknowns:
                        unk_idx += unvalidated_unknowns
                        unvalidated_unknowns = []
            if unk_idx:
                log.warn(f'Found {len(unk_idx)} unknown tokens "{[tokens[n] for n in unk_idx]}"')
    if len(tokens) > max_size:
        overflow_tokens = tokens[max_size:]
        overflow_text = ''.join(overflow_tokens)
        overflow_text = overflow_text.replace('</w>', ' ')
        log.warn(f"[bold yellow]Warning! Conditioning overflow...[/bold yellow] [red]Lost text:[/red] {overflow_text}")


""" Conditionings """

def get_cond_and_uncond(prompt, unconditional_prompt, batch_size, model, **kwargs):
    if not 'conditioning_transforms' in kwargs:
        kwargs['conditioning_transforms'] = None
    conditioning = build_conditioning(model, prompt, kwargs['conditioning_transforms'])
    log.debug('conditioning %s %s', conditioning.size(), conditioning)
    # Layer 2D conditionings to a 3D Array Cube
    conditioning = batch_conditioning(conditioning, batch_size)

    if 'guidance_scale' in kwargs and kwargs['guidance_scale'] <= 1.0:
        return conditioning, torch.zeros_like(conditioning)

    if not unconditional_prompt:
        unconditional_prompt = ""
    if not 'unconditional_transforms' in kwargs:
        rndSeed = None
        if rndSeed is None:
            rndSeed = np.random.randint(low=0, high=2**32 - 1, dtype=np.uint32)
        if unconditional_prompt.startswith('**'): # Random float to tensors.
            if len(unconditional_prompt) > 2:
                rndSeed = int(unconditional_prompt[2:])
            max_tokens_len = conditioning.size(dim=1)
            max_dimensions_len = conditioning.size(dim=2)
            log.info(f'Unconditional conditioning set to random, returning {max_tokens_len}x{max_dimensions_len} random tensors.')
            unconditional_conditioning = get_random_conditioning(max_tokens_len, max_dimensions_len, model.device, rndSeed)
            log.debug('unconditional_conditioning %s %s', unconditional_conditioning.size(), unconditional_conditioning)
            print_conditioning_tensor('**', unconditional_conditioning)
        elif unconditional_prompt.startswith('*'): # Random tokens to tensors.
            if len(unconditional_prompt) > 1:
                rndSeed = int(unconditional_prompt[1:])
            max_len = get_token_length(model.cond_stage_model)
            vocab_size = get_vocabulary_size(model.cond_stage_model)
            rng = np.random.default_rng(seed=rndSeed)
            # First number from fixed seed is how much space to keep free.
            # Set low to remove at least the space taken by start and stop tokens.
            # Up to a single token.
            max_len -= rng.integers(low=2, high=max_len - 1)
            log.info('** Unconditional conditioning set to random **')
            log.info(f'Returning new prompt text of {max_len} tokens using random sequence {rndSeed} with a vocabulary of {vocab_size} tokens.')
            ids = rng.integers(vocab_size, size=max_len, dtype=np.uint16)
            unconditional_prompt = decode_ids(model.cond_stage_model, ids)
            log.info(f'Unconditional prompt text replaced by "{unconditional_prompt}"')
            unconditional_conditioning = build_conditioning(model, unconditional_prompt, None)
            print_conditioning_tensor('*', unconditional_conditioning)
        else:
            unconditional_conditioning = build_conditioning(model, unconditional_prompt, None)
    else:
        unconditional_conditioning = build_conditioning(model, unconditional_prompt, kwargs['unconditional_transforms'])
    # Layer 2D conditionings to a 3D Array
    unconditional_conditioning = batch_conditioning(unconditional_conditioning, batch_size)

    log.debug('Batched unconditional_conditioning %s %s', unconditional_conditioning.size(), unconditional_conditioning)
    return conditioning, unconditional_conditioning

def build_conditioning(model, baseText, transforms):
    if baseText is None:
        baseText = ''
    if transforms == 'parse' or baseText.startswith('!'):
        baseText, transforms = parse_prompt(baseText[1:])
        log.info('Parser enabled, BasePrompt: "%s" transforms: %s', baseText, transforms)
    conditioning = model.get_learned_conditioning(baseText)[0]
    if not transforms:
        tokens, ids = encode_tokens(model.cond_stage_model, baseText)
        print_token_loss(model.cond_stage_model, tokens)
        print_text_conditioning(model, baseText, tokens, ids, conditioning)
        return conditioning
    max_size = get_token_length(model.cond_stage_model)
    tokens, ids = encode_tokens(model.cond_stage_model, baseText, wrapped=True, size=max_size)
    print_text_conditioning(model, baseText, tokens, ids, conditioning)
    return transform_conditioning(model, conditioning, ids, transforms, wrapped=True)

def transform_conditioning(model, conditioning, tokens_ids, transforms, wrapped=True):
    if hasattr(conditioning, 'dtype'):
        dtype = conditioning.dtype
    else:
        dtype = torch.float16
    conditioning = to_tensor(conditioning, model.device, torch.float32)
    log.debug(f'transform_conditioning {transforms} {tokens_ids} {len(conditioning)}')
    assert (len(conditioning) >= len(tokens_ids))
    orig_mean = conditioning.mean()
    orig_std = conditioning.std()
    for transform in transforms:
        text = transform['text']
        tokens, ids = encode_tokens(model.cond_stage_model, text)
        print_token_loss(model.cond_stage_model, tokens, max_size=len(tokens_ids))

        subcond = None
        if 'slerp' in transform:
            subcond = model.get_learned_conditioning(text)[0]
            subcond = to_tensor(subcond, model.device, torch.float32)
            if not wrapped:  # Full sentences are wrapped, keep the subprompt wrapped for those.
                subcond = subcond[1: len(ids) + 1]

        if 'transforms' in transform:
            log.debug(f'transform["transforms"] {transform["transforms"]} "{text}" {tokens}')
            if subcond is None:
                for i in range(len(tokens_ids)):
                    if np.array_equal(tokens_ids[i: i + len(ids)], ids):
                        subcond = conditioning[i: i + len(ids)]
            if subcond is not None:
                log.info('Applying transforms %s to tokens %s ids %s', transform['transforms'], tokens, ids)
                subcond = transform_conditioning(model, subcond, ids, transform['transforms'], wrapped=False)
                if 'slerp' not in transform:
                    # Replace current token with transformed version.
                    for i in range(len(tokens_ids)):
                        if np.array_equal(tokens_ids[i: i + len(ids)], ids):
                            conditioning[i: i + len(ids)] = subcond
                            log.info(f'{len(transform["transforms"])} transforms[{i}:{i + len(ids)}] applied to "{text}" {tokens}')
                            break
                    else:
                        log.warn('Missing token(s) "%s", can\'t apply transform!', text)
                        log.debug('Looking for %s into %s', ids, tokens_ids)
            else:
                log.warn('Missing subcond "%s", can\'t apply transform!', text)

        if 'slerp' in transform:
            slerp_range = len(subcond) if wrapped else min(len(tokens_ids), len(ids))
            log.debug('SLERP "%s" %s over range %s', text, ids, slerp_range)
            alpha = transform['slerp']
            for i in range(slerp_range):
                conditioning[i] = slerp(conditioning[i], subcond[i], alpha)
            log.info('SLERP "%s" %s, Mixed by %s', text, tokens, alpha)

        if 'weight' in transform:
            weight = transform['weight']
            for i in range(len(tokens_ids)):
                if np.array_equal(tokens_ids[i: i + len(ids)], ids):
                    for j in range(len(ids)):
                        conditioning[i + j] *= weight
                    log.info('WEIGHT "%s" %s Ids: %s Scaled %d token(s) by %s', text, tokens, ids, len(ids), weight)
                    break
            else:
                log.debug('WEIGHT Ids %s not found in prompt tokens %s', ids, tokens_ids)
                log.warn('WEIGHT Missing "%s", can\'t apply transform!', text)

    conditioning *= orig_std / conditioning.std()
    conditioning += orig_mean - conditioning.mean()
    return to_tensor(conditioning, model.device, dtype)

def batch_conditioning(cond, batch_size):
    if isinstance(cond, torch.Tensor):
        return cond.repeat(batch_size, 1, 1)
    cond_batch = []
    for _ in range(batch_size):
        cond_batch.append(cond)
    cond_batch = np.vstack(cond_batch).astype(np.float32)

def print_conditioning_tensor(cond_text, cond_tensor):
    pMean = torch.mean(cond_tensor).type(torch.FloatTensor)
    pStd = torch.std(cond_tensor).type(torch.FloatTensor)
    pMin = torch.min(cond_tensor).type(torch.FloatTensor)
    pMax = torch.max(cond_tensor).type(torch.FloatTensor)
    log.debug(f'{cond_text}: Min:{pMin}, Mean:{pMean}, Max:{pMax}, Deviation:{pStd}')

def print_text_conditioning(model, text, tokens, ids, conditioning):
    if hasattr(model.cond_stage_model, 'tokenizer') and model.cond_stage_model.tokenizer:
        tokenizer = model.cond_stage_model.tokenizer
        if len(tokens) > 0 and tokens[0] == tokenizer.bos_token:
            tokens = tokens[1:]
            #ids = ids[1:]
        tokens = [t for idx, t in enumerate(tokens) if t != (tokenizer.pad_token if idx < len(tokens) - 1 and hasattr(tokenizer, 'pad_token') else tokenizer.eos_token)]
        #ids = ids[:len(tokens)]

    log.info('tokens %s len %d', tokens, len(tokens))
    log.debug('ids %s len %d', ids, len(ids))
    print_conditioning_tensor(text, conditioning)
    #if ' ' not in text:
    #    return
    #for word in text.split():
    #    cond = model.get_learned_conditioning(word)
    #    print_conditioning_tensor(word, cond)

def get_random_conditioning(nbr_tokens, nbr_dimensions, device, seed=None, alpha=0.25):
    generator = torch.Generator(device=device)
    if seed is None:
        seed = generator.seed()
    log.info('Generating random conditioning from seed %d', seed)
    rnd_cond = torch.rand((nbr_tokens - 2, nbr_dimensions), generator=generator.manual_seed(seed), device=device)
    rnd_cond = torch.sub(torch.mul(rnd_cond, 1.5), 0.5)
    rnd_cond = torch.mul(rnd_cond, 3 * alpha)
    #rnd_cond /= rnd_cond.norm(dim=-1, keepdim=True)
    #rnd_cond /= (rnd_cond.norm(dim=-1, keepdim=True) * 0.25)
    #rnd_cond = torch.randn((nbr_tokens - 2, nbr_dimensions), generator=generator.manual_seed(seed), device=device)
    rnd_cond = F.pad(input=rnd_cond, pad=(0, 0, 1, 1), mode='constant', value=0)
    return rnd_cond

def get_random_latent(in_channels, height, width, device, seed=None):
    generator = torch.Generator(device=device)
    if seed is None:
        seed = generator.seed()
    log.info('Generating random image latent from seed %d', seed)
    return torch.randn((1, in_channels, height // 8, width // 8), generator=generator.manual_seed(seed), device=device)

def cosine_similarity(model, text_tokens, images):
    image_input = torch.tensor(np.stack(images))
    #text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    return similarity
    #torch.cosine_similarity(text_features, image_features)

def slerp(v1, v2, t, DOT_THR=0.9995, to_cpu=False, zdim=-1):
    """Spherical linear interpolation for pytorch tensors interpolating `v1` to `v2` with scale of `t`.
    Args:
        t (float/tensor): Float value between 0.0 and 1.0
        v0 (tensor): Starting vector
        v1 (tensor): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as colineal. Not recommended to alter this.
    Returns:
        v2 (tensor): Interpolation vector between v0 and v1

    `DOT_THR` determines when the vectors are too close to parallel.
        If they are too close, then a regular linear interpolation is used.

    `to_cpu` is a flag that optionally computes SLERP on the CPU.
        If the input tensors were on a GPU, it moves them back after the computation.  

    `zdim` is the feature dimension over which to compute norms and find angles.
        For example: if a sequence of 5 vectors is input with shape [5, 768]
        Then `zdim = 1` or `zdim = -1` computes SLERP along the feature dim of 768.

    Theory Reference:
    https://splines.readthedocs.io/en/latest/rotation/slerp.html
    PyTorch reference:
    https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
    Numpy reference: 
    https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
    Src:
    https://enzokro.dev/blog/posts/2022-11-16-pytorch-slerp/

    https://www.inference.vc/high-dimensional-gaussian-distributions-are-soap-bubble/
    """

    if to_cpu: # check if we need to move to the cpu
        orig_device = v1.device
        v1, v2 = v1.to('cpu'), v2.to('cpu')

    # take the dot product between normalized vectors
    v1_norm = v1 / torch.norm(v1, dim=zdim, keepdim=True)
    v2_norm = v2 / torch.norm(v2, dim=zdim, keepdim=True)
    dot = (v1_norm * v2_norm).sum(zdim)

    # if the vectors are too close, return a simple linear interpolation
    if (torch.abs(dot) > DOT_THR).any():
        log.debug('v1 and v2 close to parallel, using linear interpolation instead.')
        #res = (1 - t) * v1 + t * v2
        res = torch.lerp(v1, v2, t)

    else: # else apply SLERP
        # compute the angle terms we need
        theta = torch.acos(dot)
        theta_t = theta * t
        sin_theta = torch.sin(theta)
        sin_theta_t = torch.sin(theta_t)

        # compute the sine scaling terms for the vectors
        s1 = torch.sin(theta - theta_t) / sin_theta
        s2 = sin_theta_t / sin_theta

        # interpolate the vectors
        res = (s1.unsqueeze(zdim) * v1) + (s2.unsqueeze(zdim) * v2)

    if to_cpu: # check if we need to move them back to the original device
        res.to(orig_device)

    return res

#loss.mean()
def spherical_distance(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    l = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()
    return l
