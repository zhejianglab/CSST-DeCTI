from .dncnn import DnCNN
from .dectiabla import DeCTIAbla


def make_model(
    model_name="DeCTIAbla",
    data_length=9232,
    window_size=64,
    abla_rpe=1,
    abla_ape=1,
    abla_residual=1,
    abla_patch_size=1,
):

    name = model_name.lower()
    if name == "dncnn":
        model = DnCNN(in_nc=1, out_nc=1, nc=96, nb=20, act_mode="bR")
        params = {"model_name": name}

    elif name == "dectiabla":
        model = DeCTIAbla(
            seq_len=data_length,
            patch_size=abla_patch_size,
            in_chans=1,
            window_size=window_size,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=96,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2.0,
            ape=abla_ape,
            rpe=abla_rpe,
            residual=abla_residual,
        )
        params = {"model_name": name, "data_length": data_length, "window_size": window_size, "abla_rpe": abla_rpe,
                  "abla_ape": abla_ape, "abla_residual": abla_residual, "abla_patch_size": abla_patch_size}

    else:
        raise Exception("Invalid model name: {}".format(model_name))

    return model, params
