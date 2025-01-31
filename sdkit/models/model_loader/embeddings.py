import os

from sdkit import Context
from sdkit.utils import load_tensor_file, log

from sdkit import Context

EMBEDDING_TYPES = {768: "SD1", 1024: "SD2", 1280: "SDXL"}


def load_model(context: Context, **kwargs):
    embeddings_path = context.model_paths["embeddings"]
    embeddings_paths = embeddings_path if isinstance(embeddings_path, list) else [embeddings_path]

    model = context.models["stable-diffusion"]
    pipe = model["default"]
    sd_type = model["type"]

    components = [(pipe.tokenizer, pipe.text_encoder)]
    if sd_type == "SDXL":
        components.append((pipe.tokenizer_2, pipe.text_encoder_2))

    embeddings, embedding_tokens = get_embeddings_to_load(pipe, embeddings_paths, sd_type)

    # remove the cpu offload hook, if necessary
    is_cpu_offloaded = hasattr(pipe.text_encoder, "_hf_hook")
    if is_cpu_offloaded:
        remove_hooks(components)

    # load the embeddings
    try:
        load_embeddings(embeddings, embedding_tokens, components)
    finally:
        # reattach the cpu offload hook, if necessary
        if is_cpu_offloaded:
            attach_hooks(context, components)

    return {}


def unload_model(context: Context, **kwargs):
    pass


def get_embeddings_to_load(pipe, embeddings_paths, sd_type):
    embeddings = []
    embedding_tokens = []

    vocab = pipe.tokenizer.get_vocab()

    for path in embeddings_paths:
        token = get_embedding_token(path.lower())
        if token in vocab:
            continue

        embedding = load_tensor_file(path)
        embedding = get_embedding(embedding)
        if not embedding:
            raise RuntimeError(f"Sorry, could not load {path}. Unknown embedding model type!")

        embedding_type = EMBEDDING_TYPES.get(embedding[-1].shape[1])
        if embedding_type != sd_type:
            raise RuntimeError(
                f"Sorry, could not load {path}. You're trying to use a {embedding_type} embedding model with a {sd_type} Stable Diffusion model. They're not compatible, please use a compatible model!"
            )

        embeddings.append(embedding)
        embedding_tokens.append(token)

    return embeddings, embedding_tokens


def load_embeddings(embeddings, embedding_tokens, components):
    for embedding, token in zip(embeddings, embedding_tokens):
        for i in range(len(embedding)):
            embed = embedding[i]
            tokenizer, text_encoder = components[i]

            is_multi_vector = len(embed.shape) > 1 and embed.shape[0] > 1
            if is_multi_vector:
                tokens = [token] + [f"{token}_{i}" for i in range(1, embed.shape[0])]
                embeds = [e for e in embed]  # noqa: C416
            else:
                tokens = [token]
                embeds = [embed]

            tokenizer.add_tokens(tokens)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            text_encoder.resize_token_embeddings(len(tokenizer))
            for token_id, e in zip(token_ids, embeds):
                text_encoder.get_input_embeddings().weight.data[token_id] = e

            log.info(f"Loaded embedding for token {token} in text_encoder {i+1}")


def remove_hooks(components):
    from accelerate.hooks import remove_hook_from_module

    for _, te in components:
        remove_hook_from_module(te, recurse=True)


def attach_hooks(context, components):
    from accelerate import cpu_offload

    for _, te in components:
        cpu_offload(te, context.torch_device, offload_buffers=len(te._parameters) > 0)


def get_embedding(embedding):
    if "emb_params" in embedding:
        return [embedding["emb_params"]]
    elif "<concept>" in embedding:
        return [embedding["<concept>"]]
    elif "string_to_param" in embedding:
        for trained_token in embedding["string_to_param"]:
            return [embedding["string_to_param"][trained_token]]
    elif "clip_l" in embedding and "clip_g" in embedding:
        return [embedding["clip_l"], embedding["clip_g"]]


def get_embedding_token(filename):
    return os.path.basename(filename).split(".")[0].replace(" ", "_")
