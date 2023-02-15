# Modified version, originally contributed by c0bra5 - https://github.com/cmdr2/stable-diffusion-ui/pull/619
# which was a cut down version of https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/c9a2cfdf2a53d37c2de1908423e4f548088667ef/modules/hypernetworks/hypernetwork.py

import inspect

import torch


class HypernetworkModule(torch.nn.Module):
    multiplier = 0.5
    activation_dict = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }
    activation_dict.update(
        {
            cls_name.lower(): cls_obj
            for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation)
            if inspect.isclass(cls_obj) and cls_obj.__module__ == "torch.nn.modules.activation"
        }
    )

    def __init__(
        self,
        dim,
        state_dict=None,
        layer_structure=None,
        activation_func=None,
        weight_init="Normal",
        add_layer_norm=False,
        use_dropout=False,
        activate_output=False,
        last_layer_dropout=False,
        model=None,
        device="cuda",
    ):
        super().__init__()

        self.model = model

        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"

        linears = []
        for i in range(len(layer_structure) - 1):
            # Add a fully-connected layer
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i + 1])))

            # Add an activation func except last layer
            if (
                activation_func == "linear"
                or activation_func is None
                or (i >= len(layer_structure) - 2 and not activate_output)
            ):
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())
            else:
                raise RuntimeError(f"hypernetwork uses an unsupported activation function: {activation_func}")

            # Add layer normalization
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i + 1])))

            # Add dropout except last layer
            if use_dropout and (i < len(layer_structure) - 3 or last_layer_dropout and i < len(layer_structure) - 2):
                linears.append(torch.nn.Dropout(p=0.3))

        self.linear = torch.nn.Sequential(*linears)

        self.fix_old_state_dict(state_dict)
        self.load_state_dict(state_dict)

        self.to(device)

    def fix_old_state_dict(self, state_dict):
        changes = {
            "linear1.bias": "linear.0.bias",
            "linear1.weight": "linear.0.weight",
            "linear2.bias": "linear.1.bias",
            "linear2.weight": "linear.1.weight",
        }

        for fr, to in changes.items():
            x = state_dict.get(fr, None)
            if x is None:
                continue

            del state_dict[fr]
            state_dict[to] = x

    def forward(self, x: torch.Tensor):
        return x + self.linear(x) * self.model["hypernetwork_strength"]


def apply_hypernetwork(hypernetwork, attention_context, layer=None):
    hypernetwork_layers = hypernetwork.get(attention_context.shape[2], None)

    if hypernetwork_layers is None:
        return attention_context, attention_context

    if layer is not None:
        layer.hyper_k = hypernetwork_layers[0]
        layer.hyper_v = hypernetwork_layers[1]

    attention_context_k = hypernetwork_layers[0](attention_context)
    attention_context_v = hypernetwork_layers[1](attention_context)
    return attention_context_k, attention_context_v


def override_attention_context_kv(hypernetwork_model):
    import sdkit.models.model_loader.stable_diffusion.optimizations as sd_model_optimizer

    def get_context_kv(attention_context):
        if hypernetwork_model is None:
            return attention_context, attention_context
        else:
            return apply_hypernetwork(hypernetwork_model, attention_context)

    sd_model_optimizer.get_context_kv = get_context_kv
