"""Utility functions for posterior inference"""
from tensorflow_probability import edward2 as ed


def make_value_setter(**model_kwargs):
    """Creates a value-setting interceptor for VI under Edward2."""

    def set_values(f, *args, **kwargs):
        """Sets random variable values to its aligned value."""
        name = kwargs.get("name")
        if name in model_kwargs:
            kwargs["value"] = model_kwargs[name]
        return ed.interceptable(f)(*args, **kwargs)

    return set_values

