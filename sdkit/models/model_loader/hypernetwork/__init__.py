"""
    Hypernetwork model loader.
"""
import traceback

from sdkit import Context
from sdkit.utils import log, load_tensor_file
from sdkit.models.model_loader.hypernetwork.hypernetwork import \
    override_attention_context_kv, HypernetworkModule


def load_model(context: Context, **__) -> dict[str, int] | None:
    """
        Load the model.
    """
    model_path = context.model_paths.get('hypernetwork')

    try:
        state_dict = load_tensor_file(model_path)

        layer_structure = state_dict.get('layer_structure', [1, 2, 1])
        activation_func = state_dict.get('activation_func', None)
        weight_init = state_dict.get('weight_initialization', 'Normal')
        add_layer_norm = state_dict.get('is_layer_norm', False)
        use_dropout = state_dict.get('use_dropout', False)
        activate_output = state_dict.get('activate_output', True)
        last_layer_dropout = state_dict.get('last_layer_dropout', False)

        layers = {'hypernetwork_strength': 0}
        for size, sd in state_dict.items():
            if type(size) == int:
                layers[size] = (
                    HypernetworkModule(
                        size, sd[i], layer_structure, activation_func,
                        weight_init, add_layer_norm, use_dropout,
                        activate_output, last_layer_dropout=last_layer_dropout,
                        model=layers, device=context.device)
                    for i in range(0, 1)
                )

        override_attention_context_kv(layers)

        return layers
    except:
        log.error(traceback.format_exc())
        log.error('Could not load hypernetwork: %s', model_path)


def unload_model(_: Context, **__) -> None:
    """
        Unload the model.
    """
    override_attention_context_kv(None)
