import torch
from transformers import LlamaForCausalLM
import typer
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

from hydragen.tp import apply_tp
from hydragen.utils import set_rank, set_world_size


def main(model_name: str, outdir: Path, num_splits: int = 8):
    """
    Download huggingface model that uses the `LlamaForCausalLM` class
    in format that can be loaded for running with tensor paralellism.

    Args:
        model_name: huggingface name for model
        outdir: where to store model shards
        num_splits: number of gpus used for tensor parallelism
    """

    outdir.mkdir(exist_ok=True, parents=True)

    set_world_size(num_splits)

    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="cpu"
    )

    for i in tqdm(range(num_splits)):
        set_rank(i)

        model_split = deepcopy(model)
        apply_tp(model_split.model)

        with open(outdir / f"{i}.pt", "wb") as f:
            torch.save(model_split.state_dict(), f)


if __name__ == "__main__":
    typer.run(main)
