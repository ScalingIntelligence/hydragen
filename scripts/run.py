from transformers import AutoTokenizer
from typing import Optional, List
import typer
import torch
from pathlib import Path

from hydragen.utils import dtype_map, maybe_init_dist, local_print
from hydragen.tp import from_pretrained_tp
from hydragen.llama import HydragenLlamaForCausalLM


def main(
    pretrained_name: str = "princeton-nlp/Sheared-LLaMA-1.3B",
    prompts: list[str] = [
        "Harry Potter is a character.",
        "He is a wizard.|His best friend is Hermione.",
        "He went to school at|He is the main character in|She is known for her|Played by the actress",
    ],
    num_return_sequences: int = 1,
    max_new_tokens: int = 16,
    device: str = "cuda",
    dtype: str = "bfloat16",
    graph: bool = True,
    tp_path: Optional[Path] = None,
    temperature: float = 1.0,
    seed: int = 42,
):
    """
    Simple script to inference models using Hydragen.
    """

    rank = maybe_init_dist()
    use_tp = rank is not None

    split_prompts = [prompt.split("|") for prompt in prompts]

    for i in range(len(split_prompts) - 1):
        assert (
            len(split_prompts[i + 1]) % len(split_prompts[i]) == 0
        ), "Number of prompts in each level must be evenly divided by number of prompts in previous level"

    dtype = dtype_map[dtype]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    local_print("Loading model...")
    if use_tp:
        assert tp_path is not None
        model = from_pretrained_tp(pretrained_name, tp_path, dtype)
    else:
        model = HydragenLlamaForCausalLM.from_pretrained(
            pretrained_name, torch_dtype=dtype, device_map=device
        )
    local_print("Done loading model!")

    torch.manual_seed(seed)

    def get_model_input(prompts: str, add_special_tokens: bool):
        encoded_prompts = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=add_special_tokens,
        )
        prompt_ids = encoded_prompts["input_ids"].to(device)
        prompt_attention_mask = encoded_prompts["attention_mask"].to(device)
        return prompt_ids, prompt_attention_mask

    tokenized_prompts = [
        get_model_input(prompt, add_special_tokens=(i == 0))
        for i, prompt in enumerate(split_prompts)
    ]

    input_ids = [prompt[0] for prompt in tokenized_prompts]
    sequence_lengths = [prompt[1].sum(1) for prompt in tokenized_prompts]

    if num_return_sequences > 1:
        shared_batch_sizes = [ids.shape[0] for ids in input_ids]
        shared_seq_lengths = [ids.shape[1] for ids in input_ids]
    else:
        shared_batch_sizes = [ids.shape[0] for ids in input_ids[:-1]]
        shared_seq_lengths = [ids.shape[1] for ids in input_ids[:-1]]

    unique_batch_size = input_ids[-1].shape[0] * num_return_sequences
    unique_seq_len = max_new_tokens

    model.setup_caches(
        max_unique_batch_size=unique_batch_size,
        max_unique_seq_length=unique_seq_len,
        max_shared_batch_sizes=shared_batch_sizes,
        max_shared_seq_lengths=shared_seq_lengths,
    )

    model.graph(graph)

    new_ids = model.generate(
        input_ids=input_ids,
        seq_lens=sequence_lengths,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    local_print("Completions:")

    local_print(
        tokenizer.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


if __name__ == "__main__":
    typer.run(main)
