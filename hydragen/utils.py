import torch
from typing import List, Optional

from pathlib import Path

import yaml

import torch.distributed as dist

import os


def rdiff(a, b, eps=1e-8):
    diff = (a - b).abs()
    return 2 * diff / (a.abs() + b.abs() + eps)


def mean(x):
    return sum(x) / len(x)


def variance(x):
    return sum((t - mean(x)) ** 2 for t in x) / len(x)


def std(x):
    return variance(x) ** 0.5


dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def dataclass_to_dict(obj) -> dict:
    """
    Converts a dataclass to a dictionary. Will recurse through
    lists, dicts, and nested dataclasses.
    """

    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def dict_to_dataclass(cls, dct: dict):
    # Handle Optional
    if getattr(cls, "_name", None) == "Optional":
        if dct is None:
            return None
        else:
            return dict_to_dataclass(cls.__args__[0], dct)

    if hasattr(cls, "__dataclass_fields__"):
        # Use __dataclass_fields__ to get the type
        return cls(
            **{
                k: dict_to_dataclass(cls.__dataclass_fields__[k].type, v)
                for k, v in dct.items()
            }
        )
    elif hasattr(cls, "__origin__") and cls.__origin__ in (list, List):  # List type
        return [dict_to_dataclass(cls.__args__[0], item) for item in dct]
    else:
        return dct


def load_yaml(path: Path):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.CLoader)

    return data


def save_yaml(path: Path, data, sort_keys=True):
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)


def get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def set_rank(rank: int):
    os.environ["LOCAL_RANK"] = str(rank)


def is_local():
    return get_rank() == 0


def local_break():
    if is_local():
        breakpoint()
    dist.barrier()


def local_print(*args, **kwargs):
    if is_local():
        print(*args, **kwargs)


def get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def set_world_size(world_size: int):
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)


def maybe_init_dist() -> Optional[int]:
    try:
        # provided by torchrun
        rank = get_rank()
        world_size = get_world_size()

        if world_size < 2:
            # too few gpus to parallelize, tp is no-op
            return None
    except KeyError:
        # not run via torchrun, no-op
        return None

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return rank
