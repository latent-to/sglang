# Cacheon DeepSeek-V4.1-Flash TP2 baseline

This branch targets the NVIDIA NVFP4 checkpoint on two B300 GPUs, with four or
sixteen concurrent interactive sessions. It starts from SGLang
`d32d8fa0d1693189fda6727dd1b02caa3d26e367`, not moving upstream main.

## Changes

- Enable the existing TP4 fused attention/MoE hyper-connection boundaries and
  custom all-reduce push plane at TP2.
- Apply upstream PR 42654's routed NVFP4 deferred finalize and PR 43070's mHC
  pipeline-stage improvement.
- Parameterize fused inverse-RoPE/WO-A for four local attention groups and admit
  TP2's shape to the strided WO-A path.
- Retain TRT-LLM attention, breakable prefill graphs, mega-mHC prefill, and
  prefill/decode interval 4 together. Preserve BF16 attention output at the
  breakable graph boundary when the query is FP8. Full prefill graphs remain
  rejected with this attention backend.
- Widen fused Q-lora RMSNorm/MXFP8 to 64 rows, and integrate the winning Cacheon
  proposal `b14339de697e77fd607719ca6c7ca61fd83c2e6a3517e57ce0ec4fae5ba9787f`'s
  routed-expert pruning at the target model's eager and captured MoE call sites.
  Do not replace nightly's model/attention implementation with the proposal's
  older SGLang 0.5.21 copies.

**Pruning changes model numerics.** It drops experts selected by only one row,
up to 40% of that row's routed weight, without renormalizing. It is
batch-dependent. Hash-routed layers, the draft model, and prefill are not pruned.
Historical C24 qualification does not establish quality at C4/C16; this branch
requires separate measurements at those loads.

## Build

From this source revision, with the pinned parent image already available:

```bash
REVISION=$(git rev-parse HEAD)
docker build --network=none --build-arg SOURCE_REVISION="$REVISION" \
  -f docker/Dockerfile.cacheon-dsv41-tp2 -t latent-sglang:dsv41-tp2 .
docker image inspect latent-sglang:dsv41-tp2 --format '{{.Id}}'
```

The Dockerfile pins the parent by digest and copies only the changed source
files. It does not upgrade dependencies or install a second serving stack.

## Serve

Set `MODEL_DIR`, `TOKENIZER_DIR`, `CACHE_DIR`, `GPU_DEVICES` (two device indices),
and `PORT` for the deployment. The model and tokenizer are local read-only
inputs; the per-engine runtime cache must be writable. Do not share a writable
cache with an engine being measured.

```bash
mkdir -p "${CACHE_DIR:?}"
docker run --rm --init --network=host --ipc=private --shm-size=128g \
  --ulimit memlock=-1:-1 --memory=1t --memory-swap=1t \
  --gpus "\"device=${GPU_DEVICES:?}\"" \
  --mount "type=bind,src=${MODEL_DIR:?},dst=/model,readonly" \
  --mount "type=bind,src=${TOKENIZER_DIR:?},dst=/tokenizer,readonly" \
  --mount "type=bind,src=${CACHE_DIR},dst=/cache" \
  -e HOME=/cache/home -e CUDA_CACHE_PATH=/cache/cuda \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer -e HF_HOME=/cache/huggingface \
  -e SGLANG_CACHE_DIR=/cache/sglang -e TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor -e TRITON_CACHE_DIR=/cache/triton \
  -e TRITON_HOME=/cache/triton-home -e XDG_CACHE_HOME=/cache/xdg \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 -e SGLANG_TIMEOUT_KEEP_ALIVE=900 \
  -e SGLANG_DEFAULT_THINKING=1 -e SGLANG_DSV41_REASONING_EFFORT=high \
  -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 \
  -e SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank \
  -e SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=1 \
  --entrypoint python3 latent-sglang:dsv41-tp2 -m sglang.launch_server \
  --model-path /model --tokenizer-path /tokenizer --served-model-name dsv41 \
  --host 127.0.0.1 --port "${PORT:?}" --trust-remote-code \
  --tp-size 2 --ep-size 1 --mem-fraction-static 0.8 --random-seed 7 \
  --chunked-prefill-size 16384 --max-running-requests 32 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --cuda-graph-max-bs-decode 64 --swa-prefix-tails 1024 \
  --prefill-decode-interval 4 --weight-loader-prefetch-checkpoints \
  --watchdog-timeout 3600 --cuda-graph-backend-prefill breakable \
  --cuda-graph-max-bs-prefill 16384 --dsv4-attn-backend trtllm
```

The inherited prefill-graph sequence-length envelope is 16K. Longer-context
prefill uses the upstream eager path; a 16K graph token bucket does not claim
that every long-context prefill is captured. Model load, JIT, autotuning, and
both target/draft graph captures must finish before measurement.

## Evidence scope

The serving experiment uses the same fixed multi-turn workload for each engine,
a discarded C16 warmup, then C4 and C16 with cache flushes. It records complete
turn counts, errors, per-user decode rate, TTFT, and total elapsed work. Quality
uses the upstream GSM8K scorer on 200 fixed held-out test questions at each
concurrency, retaining all replies and rejecting transport errors/truncation.
This is a development serving baseline, not a mainnet qualification, settlement,
or production cutover. Those authorities remain separate.
