from typing import Dict
from collections import OrderedDict


def adapt_checkpoint(state_dict: OrderedDict, remove: str = "module.", replace: str = "") -> OrderedDict:
    new_dict = OrderedDict()
    for key, weight in state_dict.items():
        new_key = key.replace(remove, replace)
        new_dict[new_key] = weight
    return new_dict


def adapt_checkpoint_dict(state_dict: OrderedDict, remove_replace: Dict[str, str]) -> OrderedDict:
    new_dict = OrderedDict()
    for key, weight in state_dict.items():
        new_key = remove_replace[key]
        new_dict[new_key] = weight
    return new_dict
