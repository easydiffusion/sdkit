from sdkit import Context

filters = [
    "scribble_hed",
    "softedge_hed",
    "scribble_hedsafe",
    "softedge_hedsafe",
    "depth_midas",
    "mlsd",
    "openpose",
    "openpose_face",
    "openpose_faceonly",
    "openpose_full",
    "openpose_hand",
    "scribble_pidinet",
    "softedge_pidinet",
    "scribble_pidsafe",
    "softedge_pidsafe",
    "normal_bae",
    "lineart_coarse",
    "lineart_realistic",
    "lineart_anime",
    "depth_zoe",
    "depth_leres",
    "depth_leres++",
    "shuffle",
    "canny",
    "segment",
]


def make_load_model(model_type: str):
    def load_model(context: Context, **kwargs):
        if model_type == "segment":
            from controlnet_aux import SamDetector

            model = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
        else:
            from controlnet_aux.processor import Processor

            model = Processor(model_type)

            if hasattr(model.processor, "to"):
                model.processor = model.processor.to(context.device)

        return model

    return load_model


def make_unload_model(model_type: str):
    def unload_model(context: Context, **kwargs):
        pass

    return unload_model
