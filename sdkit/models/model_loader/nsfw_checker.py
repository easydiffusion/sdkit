from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from sdkit import Context


def load_model(context: Context, **kwargs):
    model_path = "CompVis/stable-diffusion-safety-checker"

    safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

    return (safety_checker, feature_extractor)


def unload_model(context: Context, **kwargs):
    pass
