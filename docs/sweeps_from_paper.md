
# Benchmarks

Environment variables for where to save model files and benchmark results:
```bash
export MODEL_SAVE_DIR=[INSERT MODEL_SAVE_DIR HERE]
export RESULTS_SAVE_DIR=[INSERT RESULTS_SAVE_DIR HERE]
```

## End-to-end

We provide a script to benchmark our model, our model without using a shared cache (hydragen_disabled), our model while skipping the attention operation (noattention) and vLLM on synthetic data. This script allows you to specify values to sweep over for the token length of the shared prompt, token length of the generation and the number of completions.

The following commands can be used to reproduce the results in section C.1 of the paper:

Download and partition models:
```
python hydragen/make_tp_files.py codellama/CodeLlama-7b-hf $MODEL_SAVE_DIR/split8-codellama7b --num-splits 8

python hydragen/make_tp_files.py codellama/CodeLlama-13b-hf $MODEL_SAVE_DIR/split8-codellama13b --num-splits 8

python hydragen/make_tp_files.py codellama/CodeLlama-34b-hf $MODEL_SAVE_DIR/split8-codellama34b --num-splits 8
```

### Hydragen
```
torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b 32:129:x2 1024,2048,4096,8192,16256 128 --mode hydragen --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b 256:2049:x2 1024,2048,4096,8192,16256 128 --mode hydragen --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b-c256 32:129:x2 1024,2048,4096,8192,16128 256 --mode hydragen --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b-c256 256:1025:x2 1024,2048,4096,8192,16128 256 --mode hydragen --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b 32:129:x2 1024,2048,4096,8192,16256 128 --mode hydragen --tp-dir $MODEL_SAVE_DIR/split8-codellama13b --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b 256:1025:x2 1024,2048,4096,8192,16256 128 --mode hydragen --tp-dir $MODEL_SAVE_DIR/split8-codellama13b --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-13b-Instruct-hf

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b-c256 32:129:x2 1024,2048,4096,8192,16128 256 --mode hydragen --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama13b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b-c256 256:513:x2 1024,2048,4096,8192,16128 256 --mode hydragen --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-13b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama13b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b 32:129:x2 1024,2048,4096,8192,16256 128 --mode hydragen --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b 256:4097:x2 1024,2048,4096,8192,16256 128 --mode hydragen --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b-c256 32:129:x2 1024,2048,4096,8192,16128 256 --mode hydragen --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b-c256 256:2049:x2 1024,2048,4096,8192,16128 256 --mode hydragen --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b
```

### No Attention
Note, for the no attention runs, the throughput is independent of the shared sequence length so we only run with a shared sequence length of 1024.
```
torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b 32:2049:x2 1024 128 --mode noattention --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b-c256 32:1025:x2 1024 256 --mode noattention --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b 32:1025:x2 1024 128 --mode noattention --tp-dir $MODEL_SAVE_DIR/split8-codellama13b --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b-c256 32:513:x2 1024 256 --mode noattention --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama13b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b 32:4097:x2 1024 128 --mode noattention --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b-c256 32:2049:x2 1024 256 --mode noattention --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b
```

### Flash Attention
```
torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b 32 1024,2048,4096,8192,16256 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b 64 1024,2048,4096,8192 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b 128 1024,2048,4096 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b 256 1024,2048 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b-c256 32 1024,2048,4096,8192,16128 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b-c256 64 1024,2048,4096,8192 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b-c256 128 1024,2048,4096 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b-c256 256 1024 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-7b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama7b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b 32 1024,2048,4096,8192 128 --mode hydragen_noshared --tp-dir $MODEL_SAVE_DIR/split8-codellama13b --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b 64 1024,2048,4096 128 --mode hydragen_noshared --tp-dir $MODEL_SAVE_DIR/split8-codellama13b --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b 128 1024,2048 128 --mode hydragen_noshared --tp-dir $MODEL_SAVE_DIR/split8-codellama13b --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b 256 1024 128 --mode hydragen_noshared --tp-dir $MODEL_SAVE_DIR/split8-codellama13b --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b-c256 32 1024,2048,4096,8192 256 --mode hydragen_noshared --tp-dir $MODEL_SAVE_DIR/split8-codellama13b --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b-c256 64 1024,2048,4096 256 --mode hydragen_noshared --tp-dir $MODEL_SAVE_DIR/split8-codellama13b --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b-c256 128 1024,2048 256 --mode hydragen_noshared --tp-dir $MODEL_SAVE_DIR/split8-codellama13b --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b-c256 256 1024 256 --mode hydragen_noshared --tp-dir $MODEL_SAVE_DIR/split8-codellama13b --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-13b-Instruct-hf

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b 32:65:x2 1024,2048,4096,8192,16256 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b 128 1024,2048,4096,8192 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b 256 1024,2048,4096 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b 512 1024,2048 128 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b-c256 32:65:x2 1024,2048,4096,8192,16128 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b-c256 128 1024,2048,4096,8192 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b-c256 256 1024,2048,4096 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b

torchrun --standalone --nproc_per_node=8 scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b-c256 512 1024,2048 256 --mode hydragen_noshared --num-iters 10 --num-warmup 10 --model-name codellama/CodeLlama-34b-Instruct-hf --tp-dir $MODEL_SAVE_DIR/split8-codellama34b
```


### vLLM 
use --mode vllm or --mode vllm_notok depending on if [this line](https://github.com/vllm-project/vllm/blob/2e0b6e775756345aa1d39f772c186e00f8c29e92/vllm/engine/llm_engine.py#L468) in the installed vLLM is commented out
```
python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b 32:65:x2 1024,2048,4096,8192,16256 128 --mode vllm --tp 8 --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-7b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b 128:2049:x2 1024,2048,4096,8192,16256 128 --mode vllm --tp 8 --num-iters 1 --num-warmup 1 --model-name codellama/CodeLlama-7b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b-c256 32:65:x2 1024,2048,4096,8192,16128 256 --mode vllm --tp 8 --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-7b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-7b-c256 128:2049:x2 1024,2048,4096,8192,16128 256 --mode vllm --tp 8 --num-iters 1 --num-warmup 1 --model-name codellama/CodeLlama-7b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b 32:65:x2 1024,4096,8192,16256 128 --mode vllm --tp 8 --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-13b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b 128:513:x2 1024,4096,8192,16256 128 --mode vllm --tp 8 --num-iters 1 --num-warmup 1 --model-name codellama/CodeLlama-13b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b 32:1025:x2 2048 128 --mode vllm --tp 8 --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-13b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b 1024 1024,4096,8192,16256 128 --mode vllm --tp 8 --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-13b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b-c256 32:65:x2 1024,2048,4096,8192,16128 256 --mode vllm --tp 8 --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-13b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-13b-c256 128:2049:x2 1024,2048,4096,8192,16128 256 --mode vllm --tp 8 --num-iters 1 --num-warmup 1 --model-name codellama/CodeLlama-13b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b 32:65:x2 1024,2048,4096,8192,16256 128 --mode vllm --tp 8 --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-34b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b 128:4097:x2 1024,2048,4096,8192,16256 128 --mode vllm --tp 8 --num-iters 1 --num-warmup 1 --model-name codellama/CodeLlama-34b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b-c256 32:65:x2 1024,2048,4096,8192,16128 256 --mode vllm --tp 8 --num-iters 3 --num-warmup 3 --model-name codellama/CodeLlama-34b-Instruct-hf

python scripts/synth.py $RESULTS_SAVE_DIR/e2e/fsynth-34b-c256 128:4097:x2 1024,2048,4096,8192,16128 256 --mode vllm --tp 8 --num-iters 1 --num-warmup 1 --model-name codellama/CodeLlama-34b-Instruct-hf
```

## Microbenchmarks

Similarly, we provide a script that benchmarks the speed of only the attention operation with and without a shared cache while varying the same parameters. The following commands can be used to reproduce the results of Figure 5 and 8.

Note that we make independant calls to the baseline script for the largest batch size/sequence length combination because we get a CUDA memory error when trying to run inference twice in the same script with this combination of sizes.
```bash
for i in base hydragen; do
    python scripts/microbenchmark.py $RESULTS_SAVE_DIR/microbenchmarks 512 1024 0:513:16 --mode $i
    python scripts/microbenchmark.py $RESULTS_SAVE_DIR/microbenchmarks 1024 2048 0:513:16 --mode $i
    python scripts/microbenchmark.py $RESULTS_SAVE_DIR/microbenchmarks 2048 4096 0:513:16 --mode $i
done

python scripts/microbenchmark.py $RESULTS_SAVE_DIR/microbenchmarks 4096 8192 0:513:16 --mode hydragen

for u in {0..512..16}; do
  python scripts/microbenchmark.py $RESULTS_SAVE_DIR/microbenchmarks 4096 8192 $u --mode base
done
```

## Needle-in-a-Haystack

The following commands can be used to reproduce the results of Figure 6. The script uses the text "War and Peace", saved in the data directory.

Download Yi-6B-200k model
```bash
python hydragen/make_tp_files.py 01-ai/Yi-6B-200K $MODEL_SAVE_DIR/split4-yi-6b-200k --num-splits 4
```

```bash
torchrun --standalone --nproc_per_node=4  scripts/needles.py --pretrained-name 01-ai/Yi-6B-200K --tp-path $MODEL_SAVE_DIR/split4-yi-6b-200k --save-dir $RESULTS_SAVE_DIR/needles --save-name=hydragen --base-prompt-string-length 50000 --num-timing-iters 10

torchrun --standalone --nproc_per_node=4  scripts/needles.py --pretrained-name 01-ai/Yi-6B-200K --tp-path $MODEL_SAVE_DIR/split4-yi-6b-200k --save-dir $RESULTS_SAVE_DIR/needles --save-name=no_attention --base-prompt-string-length 50000 --num-timing-iters 10 --disable-attention

torchrun --standalone --nproc_per_node=4  scripts/needles.py --pretrained-name 01-ai/Yi-6B-200K --tp-path $MODEL_SAVE_DIR/split4-yi-6b-200k --save-dir $RESULTS_SAVE_DIR/needles --save-name=disable_hydragen --base-prompt-string-length 50000 --num-timing-iters 10 --disable-hydragen
```
