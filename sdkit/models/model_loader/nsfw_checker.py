from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from sdkit import Context


def load_model(context: Context, **kwargs):
    model_path = "CompVis/stable-diffusion-safety-checker"
    revision = "cb41f3a270d63d454d385fc2e4f571c487c253c5"

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_path, revision=revision)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, revision=revision)

    return (safety_checker, feature_extractor)


def unload_model(context: Context, **kwargs):
    pass
