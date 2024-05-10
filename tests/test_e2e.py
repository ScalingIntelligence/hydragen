import torch

from hydragen.llama import HydragenLlamaForCausalLM, repeat_to_batch_size
from transformers import AutoModelForCausalLM

from itertools import product

from tqdm import tqdm

from dataclasses import dataclass

from copy import deepcopy
from hydragen.utils import rdiff


@dataclass
class CacheInfo:
    batch_sizes: list[int]
    seq_lens: list[int]
    padding: int = 0


def test_e2e():

    model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
    dtype = torch.float16
    device = "cuda:0"

    atol = 0.75
    rtol = 0.05

    model = HydragenLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )

    torch.manual_seed(0)

    num_completions_list = [1, 2]
    num_tokens_list = [4]
    graph_list = [False, True]

    cache_list = [
        CacheInfo(batch_sizes=[1], seq_lens=[10]),
        CacheInfo(batch_sizes=[1], seq_lens=[10], padding=3),
        CacheInfo(batch_sizes=[1, 4], seq_lens=[2, 4]),
        CacheInfo(batch_sizes=[2, 4, 8], seq_lens=[2, 4, 3]),
        CacheInfo(batch_sizes=[3, 3, 6], seq_lens=[4, 8, 2], padding=1),
        CacheInfo(batch_sizes=[1, 2], seq_lens=[256, 128]),
    ]

    for num_completions, num_tokens, cache, graph in tqdm(
        list(product(num_completions_list, num_tokens_list, cache_list, graph_list))
    ):
        # not selecting from first few special tokens since you can
        # get weird results with them, e.g. nan logits from
        # both us and huggingface
        input_ids = [
            torch.randint(3, 31000, (bs, seq_len), dtype=torch.long, device=device)
            for bs, seq_len in zip(cache.batch_sizes, cache.seq_lens)
        ]

        # start with a bos token
        input_ids[0][:, 0] = 1

        max_batch_sizes = deepcopy(cache.batch_sizes)

        max_sequence_lengths = [s + cache.padding for s in cache.seq_lens]
        if num_completions > 1:
            max_batch_sizes.append(max_batch_sizes[-1] * num_completions)
            max_sequence_lengths.append(num_tokens)
        else:
            max_sequence_lengths[-1] += num_tokens

        model.setup_caches(
            max_shared_batch_sizes=max_batch_sizes[:-1],
            max_shared_seq_lengths=max_sequence_lengths[:-1],
            max_unique_batch_size=max_batch_sizes[-1],
            max_unique_seq_length=max_sequence_lengths[-1],
        )

        model.graph(graph)

        expanded_ids = repeat_to_batch_size(input_ids, max(max_batch_sizes))

        cat_ids = torch.cat(expanded_ids, dim=1)

        ref_output = ref_model.generate(
            cat_ids,
            do_sample=False,
            max_new_tokens=num_tokens,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        ref_logits = torch.stack(ref_output["scores"], 0)
        ref_all_ids = ref_output["sequences"]

        ref_new_ids = ref_all_ids[:, cat_ids.shape[1] :]

        new_ids, new_logits_list = model.generate(
            input_ids=input_ids,
            num_return_sequences=num_completions,
            max_new_tokens=num_tokens,
            temperature=0,
            return_logits=True,
            token_overrides=ref_new_ids,
        )
        new_logits = torch.stack(new_logits_list, 0)

        logit_adiff = (new_logits - ref_logits).abs()
        logit_rdiff = rdiff(new_logits, ref_logits)

        assert (
            torch.all(logit_adiff < atol) and logit_rdiff.mean() < rtol
        ), f"Logit fail for {num_completions=}, {num_tokens=}, {cache=}, {graph=}"


def test_disable_hydragen():

    model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
    dtype = torch.float16
    device = "cuda:0"

    torch.manual_seed(0)

    model = HydragenLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )

    num_completions_list = [1, 2]
    num_tokens_list = [4]
    graph_list = [True]

    cache_list = [
        CacheInfo(batch_sizes=[1], seq_lens=[10]),
        CacheInfo(batch_sizes=[1, 4], seq_lens=[2, 4]),
    ]

    for num_completions, num_tokens, cache, graph in tqdm(
        product(num_completions_list, num_tokens_list, cache_list, graph_list)
    ):
        print(f"Trying {num_completions=}, {num_tokens=}, {cache=}, {graph=}")

        input_ids = [
            torch.randint(1, 1024, (bs, seq_len), dtype=torch.long, device=device)
            for bs, seq_len in zip(cache.batch_sizes, cache.seq_lens)
        ]

        max_batch_sizes = deepcopy(cache.batch_sizes)

        max_sequence_lengths = [s + cache.padding for s in cache.seq_lens]
        if num_completions > 1:
            max_batch_sizes.append(max_batch_sizes[-1] * num_completions)
            max_sequence_lengths.append(num_tokens)
        else:
            max_sequence_lengths[-1] += num_tokens

        model.setup_caches(
            max_shared_batch_sizes=max_batch_sizes[:-1],
            max_shared_seq_lengths=max_sequence_lengths[:-1],
            max_unique_batch_size=max_batch_sizes[-1],
            max_unique_seq_length=max_sequence_lengths[-1],
        )

        if graph:
            model.graph()

        new_ids, new_logits_list = model.generate(
            input_ids=input_ids,
            num_return_sequences=num_completions,
            max_new_tokens=num_tokens,
            temperature=0,
            return_logits=True,
        )
        new_logits = torch.stack(new_logits_list, 0)

        # Test support for FlashAttention baseline where we only support two hierarchy levels
        # prefix + many completions or prefix + suffix + 1 completion
        if (num_completions == 1 and len(cache.batch_sizes) == 2) or (
            num_completions > 1 and len(cache.batch_sizes) == 1
        ):
            disable_max_batch_sizes = deepcopy(max_batch_sizes)
            disable_max_sequence_lengths = deepcopy(max_sequence_lengths)
            disable_max_sequence_lengths[-1] += disable_max_sequence_lengths[-2]

            model.setup_caches(
                max_shared_batch_sizes=disable_max_batch_sizes[:-1],
                max_shared_seq_lengths=disable_max_sequence_lengths[:-1],
                max_unique_batch_size=disable_max_batch_sizes[-1],
                max_unique_seq_length=disable_max_sequence_lengths[-1],
            )

            if graph:
                model.graph()

            new_ids_disable_hydragen, new_logits_list_disable_hydragen = model.generate(
                input_ids=input_ids,
                num_return_sequences=num_completions,
                max_new_tokens=num_tokens,
                temperature=0,
                disable_hydragen=True,
                return_logits=True,
            )
            new_logits_disable_hydragen = torch.stack(new_logits_list_disable_hydragen)

            assert rdiff(new_logits, new_logits_disable_hydragen).mean() < 0.02


def test_disable_hierarchy():

    model_name = "princeton-nlp/Sheared-LLaMA-1.3B"
    dtype = torch.float16
    device = "cuda:0"

    torch.manual_seed(0)

    model = HydragenLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )

    num_completions_list = [2]
    num_tokens_list = [4]
    graph_list = [True]

    cache_list = [
        CacheInfo(batch_sizes=[1, 4], seq_lens=[2, 4]),
    ]

    for num_completions, num_tokens, cache, graph in tqdm(
        product(num_completions_list, num_tokens_list, cache_list, graph_list)
    ):
        print(f"Trying {num_completions=}, {num_tokens=}, {cache=}, {graph=}")

        input_ids = [
            torch.randint(1, 1024, (bs, seq_len), dtype=torch.long, device=device)
            for bs, seq_len in zip(cache.batch_sizes, cache.seq_lens)
        ]

        max_batch_sizes = deepcopy(cache.batch_sizes)

        max_sequence_lengths = [s + cache.padding for s in cache.seq_lens]
        if num_completions > 1:
            max_batch_sizes.append(max_batch_sizes[-1] * num_completions)
            max_sequence_lengths.append(num_tokens)
        else:
            max_sequence_lengths[-1] += num_tokens

        model.setup_caches(
            max_shared_batch_sizes=max_batch_sizes[:-1],
            max_shared_seq_lengths=max_sequence_lengths[:-1],
            max_unique_batch_size=max_batch_sizes[-1],
            max_unique_seq_length=max_sequence_lengths[-1],
        )

        if graph:
            model.graph()

        new_ids, new_logits_list = model.generate(
            input_ids=input_ids,
            num_return_sequences=num_completions,
            max_new_tokens=num_tokens,
            temperature=0,
            return_logits=True,
        )
        new_logits = torch.stack(new_logits_list, 0)

        # Test support for the disable hierarchy baseline, which we only support in the case
        # where we have two levels of shared prompts, and multiple completions
        if num_completions > 1 and len(cache.batch_sizes) == 2:
            disable_max_batch_sizes = deepcopy(max_batch_sizes)
            disable_max_sequence_lengths = deepcopy(max_sequence_lengths)
            disable_max_sequence_lengths[-2] += disable_max_sequence_lengths[-1]
            disable_max_sequence_lengths = disable_max_sequence_lengths[:-1]
            disable_max_batch_sizes[-2] = disable_max_batch_sizes[-1]
            disable_max_batch_sizes = disable_max_batch_sizes[:-1]

            model.setup_caches(
                max_shared_batch_sizes=disable_max_batch_sizes[:-1],
                max_shared_seq_lengths=disable_max_sequence_lengths[:-1],
                max_unique_batch_size=disable_max_batch_sizes[-1],
                max_unique_seq_length=disable_max_sequence_lengths[-1],
            )

            new_ids_disable_hier, new_logits_list_disable_hier = model.generate(
                input_ids=input_ids,
                num_return_sequences=num_completions,
                max_new_tokens=num_tokens,
                temperature=0,
                disable_hierarchy=True,
                return_logits=True,
            )
            new_logits_disable_hier = torch.stack(new_logits_list_disable_hier)

            assert rdiff(new_logits, new_logits_disable_hier).mean() < 0.02


if __name__ == "__main__":
    test_e2e()
