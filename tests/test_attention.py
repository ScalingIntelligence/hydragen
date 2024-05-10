import torch

from hydragen.flash import flash_attention
from hydragen.attention import hydragen_attention
from hydragen.utils import rdiff


from itertools import product

from tqdm import tqdm


def test_attention():

    torch.manual_seed(0)

    qheads_list = [8]
    kvheads_list = [1, 8]
    dim_list = [128]

    # each element in this top level list is a test case,
    # corresponding to a list of seq lengths in each cache level.
    # for example, in the second test case we have two cache levels,
    # one with a single sequence of length 3, and another with
    # two sequences of length 6.
    sizes_list = [
        [[1], [10]],
        [[3], [6, 6]],
        [[3], [6, 7]],
        [[7, 7], [9, 10, 11, 4], [129, 2, 3, 4, 5, 6, 7, 128]],
        [[16384], [1, 128, 256]],
    ]

    device = "cuda:0"

    dtype = torch.float16
    atol = 2e-3
    rtol = 5e-3

    for sizes, qheads, kvheads, dim in tqdm(
        list(
            product(
                sizes_list,
                qheads_list,
                kvheads_list,
                dim_list,
            )
        )
    ):

        batch_sizes = [len(s) for s in sizes]
        final_batch_size = batch_sizes[-1]

        q = torch.randn(final_batch_size, 1, qheads, dim, device=device, dtype=dtype)

        shared_ks = []
        shared_vs = []
        shared_culens = []
        max_seqlens = []
        use_varlens = []

        for seq_lens in sizes[:-1]:
            total_len = sum(seq_lens)
            use_varlen = len(set(seq_lens)) > 1
            use_varlens.append(use_varlen)

            if use_varlen:
                sk = torch.randn(total_len, kvheads, dim, device=device, dtype=dtype)

                tensor_seq_lens = torch.tensor(
                    seq_lens, device=device, dtype=torch.int32
                )

                cumsums = tensor_seq_lens.cumsum(0)

                culens = torch.cat(
                    [torch.zeros((1,), device=device, dtype=torch.int32), cumsums]
                ).to(torch.int32)

                shared_culens.append(culens)
                max_seqlens.append(max(seq_lens))

            else:
                each_len = seq_lens[0]
                sk = torch.randn(
                    len(seq_lens), each_len, kvheads, dim, device=device, dtype=dtype
                )

                shared_culens.append(None)
                max_seqlens.append(None)

            sv = torch.randn_like(sk)

            shared_ks.append(sk)
            shared_vs.append(sv)

        final_sizes = sizes[-1]
        max_len = max(final_sizes)

        k = torch.randn(
            final_batch_size, max_len, kvheads, dim, device=device, dtype=dtype
        )
        v = torch.randn(
            final_batch_size, max_len, kvheads, dim, device=device, dtype=dtype
        )

        if len(set(final_sizes)) > 1:
            unique_seq_lens = torch.tensor(
                final_sizes, device=device, dtype=torch.int32
            )
        else:
            unique_seq_lens = None

        hy_out = hydragen_attention(
            q=q,
            k=k,
            v=v,
            shared_ks=shared_ks,
            shared_vs=shared_vs,
            shared_cu_seq_lens=shared_culens,
            shared_max_seq_lens=max_seqlens,
            use_varlens=use_varlens,
            seq_lens=unique_seq_lens,
        )

        results = []

        # for each sequence in the batch,
        # we manually assemble the full k/v
        # tensor that it uses out of the shared and
        # unique tensors.
        for i in range(final_batch_size):
            sliced_q = q[i : i + 1]
            sliced_k = k[i : i + 1]
            sliced_v = v[i : i + 1]

            if unique_seq_lens is not None:
                sliced_k = sliced_k[:, : unique_seq_lens[i]]
                sliced_v = sliced_v[:, : unique_seq_lens[i]]

            sliced_shared_ks = []
            sliced_shared_vs = []

            for j in range(len(shared_ks)):
                sk = shared_ks[j]
                sv = shared_vs[j]
                use_varlen = use_varlens[j]

                if use_varlen:
                    culens = shared_culens[j]

                    shared_batch_size = culens.shape[0] - 1
                    seqs_per_shared = final_batch_size // shared_batch_size

                    shared_index = i // seqs_per_shared

                    start = culens[shared_index]
                    end = culens[shared_index + 1]

                    sliced_shared_ks.append(sk[start:end].unsqueeze(0))
                    sliced_shared_vs.append(sv[start:end].unsqueeze(0))

                else:
                    shared_batch_size = sk.shape[0]
                    seqs_per_shared = final_batch_size // shared_batch_size

                    shared_index = i // seqs_per_shared

                    sliced_shared_ks.append(sk[shared_index].unsqueeze(0))
                    sliced_shared_vs.append(sv[shared_index].unsqueeze(0))

            # full attention state
            ks = torch.cat(sliced_shared_ks + [sliced_k], dim=1)
            vs = torch.cat(sliced_shared_vs + [sliced_v], dim=1)

            out, lse = flash_attention(sliced_q, ks, vs)

            results.append(out)

        cat_results = torch.cat(results, dim=0)

        abs_diff = torch.abs(cat_results - hy_out)
        rel_diff = rdiff(cat_results, hy_out)

        assert torch.all(abs_diff <= atol) and rel_diff.mean() <= rtol, print(
            f"Failed for {sizes=}, {qheads=}, {kvheads=}, {dim=}"
        )


if __name__ == "__main__":
    test_attention()
