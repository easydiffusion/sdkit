from sdkit.models import download_models

# download all three models (skips if already downloaded, resumes if downloaded partially)
download_models(
    models={
        "stable-diffusion": ["1.4", "1.5-pruned-emaonly-fp16"],
        "gfpgan": "1.3",
    }
)
