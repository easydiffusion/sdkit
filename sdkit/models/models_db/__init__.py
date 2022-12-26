import os
import json
from pathlib import Path

index = None

def read_models_db():
    db_path = Path(__file__).parent
    db = {}
    with open(db_path/'stable_diffusion.json') as f: db['stable-diffusion'] = json.load(f)
    with open(db_path/'gfpgan.json') as f: db['gfpgan'] = json.load(f)
    with open(db_path/'realesrgan.json') as f: db['realesrgan'] = json.load(f)
    return db

def get_model_info_from_db(quick_hash=None, model_type=None, model_id=None):
    db = read_models_db()

    if quick_hash:
        if index is None:
            rebuild_index()

        return index.get(quick_hash)
    elif model_id and model_type:
        m = db.get(model_type, {})
        return m.get(model_id)

def rebuild_index():
    global index

    db = read_models_db()
    index = {}
    for _, m in db.items():
        module_index = {info.get('quick_hash'): info for _, info in m.items()}
        index.update(module_index)
