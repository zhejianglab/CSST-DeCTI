from .dectiabla import DeCTIAbla

__all__ = ["init_model"]


def init_model(
    model_name: str = "DeCTIAbla",
    seq_len: int = 9232,
    **kwargs
):

    name = model_name.lower()
    if name.lower() == "dectiabla":
        model = DeCTIAbla(seq_len=seq_len, **kwargs)
        params = {
            "model_name": "DeCTIAbla",
            "seq_len": seq_len,
            "kwargs": kwargs,
        }

    else:
        raise Exception("Invalid model name: {}".format(model_name))

    return model, params
