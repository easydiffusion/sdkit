"""
    Models database
"""
import json
from pathlib import Path

db = None
index = None


def get_models_db():
    """
        Returns a dictionary of models info
    """
    # TODO: Use context vars instead.
    global db

    if db is not None:
        return db

    db_path = Path(__file__).parent
    db = {}
    with open(db_path/'stable_diffusion.json') as fp:
        db['stable-diffusion'] = json.load(fp)
    with open(db_path/'gfpgan.json') as fp:
        db['gfpgan'] = json.load(fp)
    with open(db_path/'realesrgan.json') as fp:
        db['realesrgan'] = json.load(fp)
    return db


def get_model_info_from_db(quick_hash=None, model_type=None, model_id=None):
    db = get_models_db()

    if quick_hash:
        if index is None:
            rebuild_index()
        if index is None:
            raise RuntimeError("Could not rebuild index")
        return index.get(quick_hash)
    elif model_id and model_type:
        m = db.get(model_type, {})
        return m.get(model_id)


def rebuild_index():
    """
        Rebuilds the index of models.
    """
    # TODO: Use contextvar instead.
    global index

    db = get_models_db()
    index = {}
    for _, m in db.items():
        module_index = {info.get('quick_hash'): info for _, info in m.items()}
        index.update(module_index)
