import torch
from itertools import product
from tqdm import tqdm

from hydragen.attention import combine_lse_triton, combine_lse_torch
from hydragen.utils import rdiff


def test_combine_lse():
    device = "cuda:0"
    bs_list = [1, 2, 3]
    seq_len_list = [1, 2, 3]
    heads_list = [1, 2, 3]
    hdim_list = [63, 64, 128, 129]

    for bs, seq_len, heads, hdim in tqdm(
        product(bs_list, seq_len_list, heads_list, hdim_list)
    ):
        out1 = torch.rand(bs, seq_len, heads, hdim, device=device)
        out2 = torch.rand_like(out1)

        lse1 = torch.rand(bs, seq_len, heads, device=device)
        lse2 = torch.rand_like(lse1)

        torch_result = combine_lse_torch([out1, out2], [lse1, lse2])
        triton_result = combine_lse_triton(out1, lse1, out2, lse2)

        rel_diff = rdiff(torch_result, triton_result)
        assert rel_diff.mean() < 0.1

