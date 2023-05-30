import json
from pathlib import Path

from sdkit.utils import log

db = None
index = None


def get_models_db():
    global db

    if db is not None:
        return db

    db_path = Path(__file__).parent
    db = {}
    with open(db_path / "stable_diffusion.json") as f:
        db["stable-diffusion"] = json.load(f)
    with open(db_path / "gfpgan.json") as f:
        db["gfpgan"] = json.load(f)
    with open(db_path / "realesrgan.json") as f:
        db["realesrgan"] = json.load(f)
    with open(db_path / "vae.json") as f:
        db["vae"] = json.load(f)
    with open(db_path / "codeformer.json") as f:
        db["codeformer"] = json.load(f)
    return db


def get_model_info_from_db(quick_hash=None, model_type=None, model_id=None):
    db = get_models_db()

    if quick_hash:
        if index is None:
            rebuild_index()
        log.debug("Checking model_db for %s", quick_hash)
        return index.get(quick_hash)
    elif model_id and model_type:
        m = db.get(model_type, {})
        return m.get(model_id)


def rebuild_index():
    global index

    db = get_models_db()
    index = {}
    for _, m in db.items():
        module_index = {info.get("quick_hash"): info for _, info in m.items()}
        index.update(module_index)
