import typer

import torch

from hydragen.attention import hydragen_attention_nopad
from hydragen.flash import flash_attention, flash_attention_seqlen

from hydragen.benchmark_utils import (
    timed_with_graphs,
    MicrobenchmarkResult,
    split_range,
)
from hydragen.utils import save_yaml, dataclass_to_dict

from typing import Optional
from tqdm import tqdm
from pathlib import Path
from itertools import product

from functools import partial

app = typer.Typer()

# 128 MB
CACHE_SIZE_BYTES = 2**27


def timed_with_graphs_and_cache_flush(
    go_func, num_iters: int, num_warmup: int, device, verbose: bool
):
    """
    Flushes the L2 cache, then calls timed_with_graphs.
    """

    cache = torch.randn(CACHE_SIZE_BYTES // 2, device=device, dtype=torch.bfloat16)

    wipe_cache = lambda: cache.random_(0, 1)

    times = timed_with_graphs(
        go_func,
        num_iters=num_iters,
        num_warmup=num_warmup,
        verbose=verbose,
        between_fn=wipe_cache,
    )

    return times


@torch.no_grad()
def go_hydragen(
    bs: int,
    num_shared: int,
    num_unique: int,
    qheads: int,
    kvheads: int,
    dim: int,
    num_iters: int,
    num_warmup: int,
    device: str,
    verbose: bool,
    unique_seq_len: Optional[int] = None,
):
    q = torch.randn(bs, 1, qheads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(bs, num_unique, kvheads, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(bs, num_unique, kvheads, dim, device=device, dtype=torch.bfloat16)

    sk = torch.randn(1, num_shared, kvheads, dim, device=device, dtype=torch.bfloat16)
    sv = torch.randn(1, num_shared, kvheads, dim, device=device, dtype=torch.bfloat16)

    if unique_seq_len is not None:
        seq_len = torch.full((bs,), unique_seq_len, device=device, dtype=torch.long)
    else:
        seq_len = None

    go = lambda: hydragen_attention_nopad(
        q,
        k,
        v,
        shared_ks=[sk],
        shared_vs=[sv],
        seq_len=seq_len,
    )

    time_func = partial(timed_with_graphs_and_cache_flush, device=device)
    times = time_func(go, num_iters=num_iters, num_warmup=num_warmup, verbose=verbose)

    return times


def go_baseline(
    bs: int,
    num_shared: int,
    num_unique: int,
    qheads: int,
    kvheads: int,
    dim: int,
    num_iters: int,
    num_warmup: int,
    device: str,
    verbose: bool,
    unique_seq_len: Optional[int] = None,
):
    total_seq_len = num_shared + num_unique

    q_tile = torch.randn(bs, 1, qheads, dim, device=device, dtype=torch.bfloat16)
    k_tile = torch.randn(
        bs, total_seq_len, kvheads, dim, device=device, dtype=torch.bfloat16
    )
    v_tile = torch.randn(
        bs, total_seq_len, kvheads, dim, device=device, dtype=torch.bfloat16
    )

    if unique_seq_len is None:
        go_og = lambda: flash_attention(q_tile, k_tile, v_tile)
    else:
        seq_len = torch.full(
            (bs,), unique_seq_len + num_shared, device=device, dtype=torch.long
        )
        go_og = lambda: flash_attention_seqlen(q_tile, k_tile, v_tile, seq_len=seq_len)

    time_func = partial(timed_with_graphs_and_cache_flush, device=device)
    times_og = time_func(
        go_og, num_iters=num_iters, num_warmup=num_warmup, verbose=verbose
    )

    return times_og


@app.command()
def sweep(
    outdir: Path,
    bs_range: str,
    num_shared_range: str,
    num_unique_range: str,
    qheads_range: str = "8",
    kvheads_range: str = "1",
    dim_range: str = "128",
    num_iters: int = 1000,
    num_warmup: int = 1000,
    mode: str = "hydragen",
    verbose: bool = False,
    unique_seq_len: Optional[int] = None,
    no_save: bool = False,
):
    
    # TODO: in order to support other devices, 
    # we need to plumb the device id through to the CUDA
    # graph capture (currently we don't and it crashes).
    # The workaround for now is to use CUDA_VISIBLE_DEVICES. 
    device = "cuda:0"

    if mode == "base":
        go_func = go_baseline

    elif mode == "hydragen":
        go_func = go_hydragen

    else:
        raise ValueError(f"Invalid run type {mode}")

    outdir.mkdir(parents=True, exist_ok=True)

    bs_list = split_range(bs_range)
    num_shared_list = split_range(num_shared_range)
    num_unique_list = split_range(num_unique_range)
    qheads_list = split_range(qheads_range)
    kvheads_list = split_range(kvheads_range)
    dim_list = split_range(dim_range)

    print("Sweeping:")
    print("bs:", bs_list)
    print("num_shared:", num_shared_list)
    print("num_unique:", num_unique_list)
    print("qheads:", qheads_list)
    print("kvheads:", kvheads_list)
    print("dim:", dim_list)

    total = (
        len(bs_list)
        * len(num_shared_list)
        * len(num_unique_list)
        * len(qheads_list)
        * len(kvheads_list)
        * len(dim_list)
    )

    with tqdm(total=total) as pbar:
        for bs, num_shared, num_unique, qheads, kvheads, dim in product(
            bs_list,
            num_shared_list,
            num_unique_list,
            qheads_list,
            kvheads_list,
            dim_list,
        ):
        
            tag = f"b{bs}_s{num_shared}_u{num_unique}_q{qheads}_k{kvheads}_d{dim}"
            fname = f"{tag}.yaml"

            outpath = outdir / mode / fname

            if outpath.exists() and not no_save:
                continue

            print(f"Running {mode} {tag}...")

            times = go_func(
                bs=bs,
                num_shared=num_shared,
                num_unique=num_unique,
                qheads=qheads,
                kvheads=kvheads,
                dim=dim,
                num_iters=num_iters,
                num_warmup=num_warmup,
                device=device,
                verbose=verbose,
                unique_seq_len=unique_seq_len,
            )
            result = MicrobenchmarkResult(
                bs=bs,
                num_shared=num_shared,
                num_unique=num_unique,
                qheads=qheads,
                kvheads=kvheads,
                dim=dim,
                times=times,
            )

            print(tag, result.mean(), result.rstd())

            if not no_save:
                outpath.parent.mkdir(parents=True, exist_ok=True)
                save_yaml(outpath, dataclass_to_dict(result))

            pbar.update(1)


if __name__ == "__main__":
    app()
