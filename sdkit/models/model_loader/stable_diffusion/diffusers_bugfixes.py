import torch
from diffusers.utils import randn_tensor
from typing import Union, Optional, List, Callable
import PIL

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_legacy import (
    preprocess_image,
    preprocess_mask,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers import StableDiffusionInpaintPipelineLegacy


# fixed in PR, pending diffusers 0.18: https://github.com/huggingface/diffusers/pull/3773
def legacy_inpainting_prepare_latents(
    self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator
):
    image = image.to(device=device, dtype=dtype)
    init_latent_dist = self.vae.encode(image).latent_dist
    init_latents = init_latent_dist.sample(generator=generator)
    init_latents = self.vae.config.scaling_factor * init_latents

    # Expand init_latents for batch_size and num_images_per_prompt
    init_latents = torch.cat([init_latents] * batch_size * num_images_per_prompt, dim=0)
    init_latents_orig = init_latents

    # add noise to latents using the timesteps
    noise = randn_tensor(init_latents.shape, generator=generator, device=device, dtype=dtype)
    init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents
    return latents, init_latents_orig, noise


# fixed in PR, pending diffusers 0.18: https://github.com/huggingface/diffusers/pull/3773
@torch.no_grad()
def legacy_inpainting_call(
    self,
    prompt: Union[str, List[str]] = None,
    image: Union[torch.FloatTensor, PIL.Image.Image] = None,
    mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
    strength: float = 0.8,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    add_predicted_noise: Optional[bool] = False,
    eta: Optional[float] = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
):
    # 1. Check inputs
    self.check_inputs(prompt, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Preprocess image and mask
    if not isinstance(image, torch.FloatTensor):
        image = preprocess_image(image)

    mask_image = preprocess_mask(mask_image, self.vae_scale_factor)

    # 5. set timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

    # 6. Prepare latent variables
    # encode the init image into latents and scale the latents
    latents, init_latents_orig, noise = self.prepare_latents(
        image, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, generator
    )

    # 7. Prepare mask latent
    mask = mask_image.to(device=device, dtype=latents.dtype)
    mask = torch.cat([mask] * batch_size * num_images_per_prompt)

    # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 9. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            # masking
            if add_predicted_noise:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise_pred_uncond, torch.tensor([t]))
            else:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))

            latents = (init_latents_proper * mask) + (latents * (1 - mask))

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    # use original latents corresponding to unmasked portions of the image
    latents = (init_latents_orig * mask) + (latents * (1 - mask))

    # 10. Post-processing
    image = self.decode_latents(latents)

    # 11. Run safety checker
    image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

    # 12. Convert to PIL
    if output_type == "pil":
        image = self.numpy_to_pil(image)

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


StableDiffusionInpaintPipelineLegacy.__call__ = legacy_inpainting_call
StableDiffusionInpaintPipelineLegacy.prepare_latents = legacy_inpainting_prepare_latents
