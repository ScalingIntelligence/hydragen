import torch
from typing import Optional, Union, List
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm


from hydragen.utils import (
    std,
    save_yaml,
    dataclass_to_dict,
    load_yaml,
    dict_to_dataclass,
)


@dataclass
class MicrobenchmarkResult:
    bs: int
    num_unique: int
    num_shared: int
    qheads: int
    kvheads: int
    dim: int
    times: list[float]

    @property
    def mean_time(self):
        return self.mean()

    def mean(self):
        return sum(self.times) / len(self.times)

    def std(self):
        return (
            sum((t - self.mean()) ** 2 for t in self.times) / len(self.times)
        ) ** 0.5

    def rstd(self):
        return self.std() / self.mean()


@dataclass
class SynthBenchmarkResult:
    bs: int
    num_unique: int
    num_shared: int
    times: list[float]
    prefill_times: Optional[list[float]] = None
    warmup_times: Optional[list[float]] = None
    prefill_warmup_times: Optional[list[float]] = None

    def mean(self):
        return sum(self.times) / len(self.times)

    def std(self):
        return (
            sum((t - self.mean()) ** 2 for t in self.times) / len(self.times)
        ) ** 0.5


@dataclass
class NeedlesBenchmarkResult:
    num_questions: int
    accuracy: float
    prefill_time: float
    times: List[float]
    warmup_times: List[float]
    accuracy_buckets: List[float]
    unique_prefill_times: List[float]
    unique_prefill_warmup_times: List[float]

    def mean(self):
        return sum(self.times) / len(self.times)

    def std(self):
        return (
            sum((t - self.mean()) ** 2 for t in self.times) / len(self.times)
        ) ** 0.5


@torch.no_grad()
def timed(
    fn,
    num_iters=50,
    num_warmup=10,
    return_type="times",
    verbose=True,
    unit: str = "ms",
    between_fn=None,
):
    warmup_times = []
    times = []
    for itr in tqdm(range(num_iters + num_warmup), desc="Timing"):

        if between_fn is not None:
            between_fn()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()

        milis = start.elapsed_time(end)
        if unit == "ms":
            time_with_unit = milis
        elif unit == "s":
            time_with_unit = milis / 1000
        elif unit == "us":
            time_with_unit = milis * 1000
        else:
            raise ValueError(f"Invalid unit: {unit}")

        if itr >= num_warmup:
            times.append(time_with_unit)
        else:
            warmup_times.append(time_with_unit)

    mean_time = sum(times) / len(times)
    std_time = std(times)

    if verbose:
        print(times)
        print(f"Time={round(mean_time, 1)}{unit}, std={round(std_time, 1)}{unit}")

    if return_type == "mean":
        return mean_time
    elif return_type == "times":
        return times
    elif return_type == "both_times":
        return times, warmup_times
    elif return_type == "all":
        return times, mean_time, std_time
    else:
        raise ValueError(f"Invalid return_type: {return_type}")


def timed_with_graphs(
    fn,
    num_iters=50,
    num_warmup=10,
    return_type="times",
    verbose=True,
    unit: str = "ms",
    between_fn=None,
):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(16):
            fn()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    go = lambda: g.replay()

    return timed(
        go,
        num_iters=num_iters,
        num_warmup=num_warmup,
        return_type=return_type,
        verbose=verbose,
        unit=unit,
        between_fn=between_fn,
    )


def save_benchmark_results(
    path: Path,
    results: List[
        Union[MicrobenchmarkResult, SynthBenchmarkResult, NeedlesBenchmarkResult]
    ],
    append: bool = False,
):
    if append:
        num_existing = len(list(path.glob("*.yaml")))
    else:
        num_existing = 0

    for i, res in enumerate(results):
        str_i = str(i + num_existing).zfill(4)
        fname = path / f"results_{str_i}.yaml"
        save_yaml(
            fname,
            dataclass_to_dict(res),
        )


def load_benchmark_results(
    path: Path,
) -> list[Union[MicrobenchmarkResult, SynthBenchmarkResult, NeedlesBenchmarkResult]]:
    paths = list(sorted(path.glob("*.yaml")))
    raw_loaded = [load_yaml(x) for x in paths]
    return [
        dict_to_dataclass(
            Union[MicrobenchmarkResult, SynthBenchmarkResult, NeedlesBenchmarkResult], x
        )
        for x in raw_loaded
    ]


def split_range(s: str):
    if ":" in s:
        start, end, step = s.split(":")

        # exponential stepping
        if step.startswith("x"):
            istart = int(start)
            iend = int(end)
            istep = int(step[1:])

            vals = []
            cur = istart

            while cur < iend:
                vals.append(cur)
                cur = cur * istep

            return vals

        else:
            return list(range(int(start), int(end), int(step)))
    else:
        return [int(x) for x in s.split(",")]
