import math
import torch
from typing import Optional
from torch.utils.data import DataLoader
import re
import torch.distributed as dist
from hydragen.utils import get_rank
from tqdm import tqdm
import sys
import typer
import os
import random
from transformers import AutoTokenizer
from pathlib import Path

from hydragen.haystack import make_needle_haystack
from hydragen.benchmark_utils import (
    NeedlesBenchmarkResult,
)
from hydragen.utils import dtype_map, maybe_init_dist, save_yaml, dataclass_to_dict
from hydragen.tp import from_pretrained_tp
from hydragen.llama import HydragenLlamaForCausalLM, SharedCacheOp

ANS_RE = re.compile(r"###(.*?)###")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example)
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


def is_correct_batched(model_completions, answers):
    return [
        is_correct(model_completion, answer)
        for model_completion, answer in zip(model_completions, answers)
    ]


def main(
    pretrained_name: str = "princeton-nlp/Sheared-LLaMA-1.3B",
    device: str = "cuda",
    dtype: str = "bfloat16",
    graph: bool = True,
    tp_path: Optional[Path] = None,
    save_dir: str = "results",
    save_name: str = "needles",
    num_few_shot: int = 5,
    disable_hydragen: bool = False,
    disable_attention: bool = False,
    base_prompt_string_length: int = 50000,
    num_timing_iters: int = 10,
    num_warmup_iters: int = 5,
    num_questions: Optional[int] = None,
    measure_unique_prefill: bool = True,
):
    rank = maybe_init_dist()
    use_tp = rank is not None

    dtype = dtype_map[dtype]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    if use_tp:
        assert tp_path is not None
        model = from_pretrained_tp(pretrained_name, tp_path, dtype)
        torch.manual_seed(9)  # needed to make sure sampling is same on each device
    else:
        model = HydragenLlamaForCausalLM.from_pretrained(
            pretrained_name, torch_dtype=dtype, device_map=device
        )

    random.seed(9)

    needle_range = (
        [2**i for i in range(1, 11)] if num_questions is None else [num_questions]
    )

    # HACK: only insert 256 questions into haystack, even when evaling more questions,
    # to avoid dramatically increasing haystack size
    haystack, needles = make_needle_haystack(
        target_context_length=base_prompt_string_length,
        num_needles=256 + num_few_shot,
    )
    random.shuffle(needles)

    few_shot_needles, needles = needles[:num_few_shot], needles[num_few_shot:]

    # HACK: repeat the needles so we can eval with a large number of questions
    needles = needles * (1 + max(needle_range) // 256)

    base_prompt = "Start of document:\n" + haystack + "\nEnd of document\n\n"
    for needle in few_shot_needles:
        base_prompt += f"Question: {needle.question}\nAnswer: ###{needle.answer}###\n\n"

    base_prompt_ids = tokenizer(
        base_prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(model.device)

    # List of Tuples, where first element is num questions,
    # second is a list of tuples incorrect answers (question, model_answer, correct_answer)
    results_per_len = []

    for num_questions in tqdm(needle_range):

        outpath = Path(save_dir + f"/{save_name}" f"/{num_questions}qs.yaml")

        if outpath.exists():
            continue

        questions = [
            "Question: " + needle.question + "\nAnswer:"
            for needle in needles[:num_questions]
        ]
        test_answers = [
            "###" + needle.answer + "###" for needle in needles[:num_questions]
        ]
        encoded_needles = tokenizer(questions, return_tensors="pt", padding=True)
        needle_ids = encoded_needles["input_ids"].to(model.device)
        needle_attention_masks = encoded_needles["attention_mask"].to(model.device)

        max_num_tokens = 10
        max_unique_seq_length = needle_ids.shape[1] + max_num_tokens
        if disable_hydragen:
            max_unique_seq_length += base_prompt_ids.shape[1]
        model.setup_caches(
            max_unique_batch_size=num_questions,
            max_unique_seq_length=max_unique_seq_length,
            max_shared_batch_sizes=[base_prompt_ids.shape[0]],
            max_shared_seq_lengths=[base_prompt_ids.shape[1]],
        )

        model.graph(graph)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        model.append_shared(base_prompt_ids)

        end.record()
        torch.cuda.synchronize()

        prefill_time = start.elapsed_time(end)

        times = []
        unique_prefill_times, unique_prefill_warmup_times = [], []

        for itr in range(num_timing_iters):

            def get_time(fn):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                out = fn()

                end.record()
                torch.cuda.synchronize()

                return out, start.elapsed_time(end)

            new_ids, time = get_time(
                lambda: model.generate(
                    input_ids=[needle_ids],
                    seq_lens=[needle_attention_masks.sum(1)],
                    num_return_sequences=1,
                    max_new_tokens=max_num_tokens,
                    temperature=0,
                    disable_hydragen=disable_hydragen,
                    disable_attention=disable_attention,
                    shared_cache_op=SharedCacheOp.PRESERVE,
                )
            )
            times.append(time)

            if measure_unique_prefill:
                _, unique_prefill_time = get_time(
                    lambda: model.generate(
                        input_ids=[needle_ids],
                        seq_lens=[needle_attention_masks.sum(1)],
                        num_return_sequences=1,
                        max_new_tokens=1,
                        temperature=0,
                        disable_hydragen=disable_hydragen,
                        disable_attention=disable_attention,
                        shared_cache_op=SharedCacheOp.PRESERVE,
                    )
                )

                unique_prefill_times.append(unique_prefill_time)

        warmup_times = times[:num_warmup_iters]
        times = times[num_warmup_iters:]
        if measure_unique_prefill:
            unique_prefill_warmup_times = unique_prefill_times[:num_warmup_iters]
            unique_prefill_times = unique_prefill_times[num_warmup_iters:]
        model_answers = tokenizer.batch_decode(new_ids)
        is_corrects = is_correct_batched(model_answers, test_answers)

        buckets = {}

        for needle, is_correct in sorted(
            zip(needles[:num_questions], is_corrects),
            key=lambda x: x[0].position_in_doc,
        ):
            key = math.floor(needle.position_in_doc * 10)
            if key not in buckets:
                buckets[key] = (0, 0)

            if is_correct:
                buckets[key] = (buckets[key][0] + 1, buckets[key][1])
            else:
                buckets[key] = (buckets[key][0], buckets[key][1] + 1)

        for key in sorted(buckets.keys()):
            buckets[key] = buckets[key][0] / (buckets[key][0] + buckets[key][1])

        if rank == 0 or rank is None:
            accuracy = sum(is_corrects) / len(is_corrects)
            print("Times:", times)

            result = NeedlesBenchmarkResult(
                num_questions=num_questions,
                accuracy=accuracy,
                prefill_time=prefill_time,
                times=times,
                warmup_times=warmup_times,
                accuracy_buckets=buckets,
                unique_prefill_warmup_times=unique_prefill_warmup_times,
                unique_prefill_times=unique_prefill_times,
            )

            outpath.parent.mkdir(parents=True, exist_ok=True)
            save_yaml(outpath, dataclass_to_dict(result))


if __name__ == "__main__":
    typer.run(main)
