# temporary patch, while waiting for PR: https://github.com/huggingface/diffusers/pull/4231
def apply_singlestep_patch():
    from diffusers.schedulers import DPMSolverSinglestepScheduler
    import numpy as np
    import torch

    old_set_timesteps = DPMSolverSinglestepScheduler.set_timesteps

    def set_timesteps_remove_duplicates(self, num_inference_steps: int, device=None):
        old_set_timesteps(self, num_inference_steps, device)

        timesteps = self.timesteps.cpu().detach().numpy().astype(np.int64)

        # when num_inference_steps == num_train_timesteps, we can end up with
        # duplicates in timesteps.
        _, unique_indices = np.unique(timesteps, return_index=True)
        timesteps = timesteps[np.sort(unique_indices)]

        self.timesteps = torch.from_numpy(timesteps).to(device)

        self.num_inference_steps = len(timesteps)

        self.order_list = self.get_order_list(self.num_inference_steps)

    DPMSolverSinglestepScheduler.set_timesteps = set_timesteps_remove_duplicates


# patch until https://github.com/huggingface/diffusers/pull/4298 is resolved
def apply_controlnet_patch():
    from diffusers.pipelines.stable_diffusion import convert_from_ckpt
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        create_unet_diffusers_config,
        convert_ldm_unet_checkpoint,
    )
    from diffusers.models import ControlNetModel
    from accelerate.utils import set_module_tensor_to_device

    def convert_controlnet_checkpoint(
        checkpoint,
        original_config,
        checkpoint_path,
        image_size,
        upcast_attention,
        extract_ema,
        use_linear_projection=None,
        cross_attention_dim=None,
    ):
        ctrlnet_config = create_unet_diffusers_config(original_config, image_size=image_size, controlnet=True)
        ctrlnet_config["upcast_attention"] = upcast_attention

        ctrlnet_config.pop("sample_size")

        if use_linear_projection is not None:
            ctrlnet_config["use_linear_projection"] = use_linear_projection

        if cross_attention_dim is not None:
            ctrlnet_config["cross_attention_dim"] = cross_attention_dim

        controlnet = ControlNetModel(**ctrlnet_config)

        # Some controlnet ckpt files are distributed independently from the rest of the
        # model components i.e. https://huggingface.co/thibaud/controlnet-sd21/
        if "time_embed.0.weight" in checkpoint:
            skip_extract_state_dict = True
        else:
            skip_extract_state_dict = False

        converted_ctrl_checkpoint = convert_ldm_unet_checkpoint(
            checkpoint,
            ctrlnet_config,
            path=checkpoint_path,
            extract_ema=extract_ema,
            controlnet=True,
            skip_extract_state_dict=skip_extract_state_dict,
        )

        for param_name, param in converted_ctrl_checkpoint.items():
            set_module_tensor_to_device(controlnet, param_name, "cpu", value=param)

        # controlnet.load_state_dict(converted_ctrl_checkpoint)

        return controlnet

    convert_from_ckpt.convert_controlnet_checkpoint = convert_controlnet_checkpoint


# waiting for https://github.com/damian0815/compel/pull/49
def apply_compel_pooled_patch():
    import torch

    from compel import Compel, EmbeddingsProvider
    from compel.embeddings_provider import EmbeddingsProviderMulti

    def build_conditioning_tensor(self, text: str) -> torch.Tensor:
        """
        Build a conditioning tensor by parsing the text for Compel syntax, constructing a Conjunction, and then
        building a conditioning tensor from that Conjunction.
        """
        conjunction = self.parse_prompt_string(text)
        conditioning, _ = self.build_conditioning_tensor_for_conjunction(conjunction)

        if self.requires_pooled:
            pooled = self.conditioning_provider.get_pooled_embeddings([text], device=self.device)
            return conditioning, pooled
        else:
            return conditioning

    def multi_get_pooled_embeddings(self, texts, attention_mask=None, device=None):
        pooled = [
            self.embedding_providers[provider_index].get_pooled_embeddings(texts, attention_mask, device=device)
            for provider_index, requires_pooled in enumerate(self.requires_pooled_mask)
            if requires_pooled
        ]

        if len(pooled) == 0:
            return None

        return torch.cat(pooled, dim=-1)

    def get_pooled_embeddings(self, texts, attention_mask=None, device=None):
        device = device or self.text_encoder.device

        token_ids = self.get_token_ids(texts, padding="max_length", truncation_override=True)
        token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)

        text_encoder_output = self.text_encoder(token_ids, attention_mask, return_dict=True)
        pooled = text_encoder_output.text_embeds

        return pooled

    Compel.build_conditioning_tensor = build_conditioning_tensor
    EmbeddingsProvider.get_pooled_embeddings = get_pooled_embeddings
    EmbeddingsProviderMulti.get_pooled_embeddings = multi_get_pooled_embeddings


apply_singlestep_patch()
apply_controlnet_patch()
apply_compel_pooled_patch()
