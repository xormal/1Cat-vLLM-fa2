# 1Cat-vLLM

> 一猫之下始终相信，V100 不该在今天的大模型浪潮里被轻易宣判“过时”。
>
> 1Cat-vLLM 是面向 **SM70 / Tesla V100** 的 vLLM 工程分支。项目围绕
> AWQ、注意力后端、长上下文稳定性、MTP 投机解码、运行时默认值和部署
> 路径做了成体系的优化，让更多现代模型场景在 V100 上真正变得可用、
> 好用、能持续部署。
>
> 我们希望把一猫之下在 V100 上的工程经验、优化成果和验证过程贡献给
> 开源社区，也欢迎继续使用 V100 的个人开发者、工作室和团队一起反馈、
> 复现和改进。

1Cat-vLLM is a **Tesla V100 / SM70** focused vLLM fork for serving modern
Qwen-class AWQ and experimental FP8 models on Volta GPUs. It integrates
TurboMind-derived SM70 kernels, a V100 FlashAttention path, runtime defaults
for long-context serving, and OpenAI-compatible API fixes for common clients.

## This fork: FA2 on SM70, int8, and what each approach bought

`xormal/1Cat-vLLM-fa2`, branch `fa2-sm70-port`, adds its own SM70 attention backend
(`FA2_SM70`, kernels in [xormal/V100-SM_70-Flash-Attn-2-v1](https://github.com/xormal/V100-SM_70-Flash-Attn-2-v1)),
GPTQ int8 bodies, and a speculative-decoding stack. The reference deployment is
**Qwen3.8-27B in GPTQ int8 (+12-bit attention weights) on 2 x V100-SXM2-32GB, TP=2 over
NVLink, 262 144-token int8 KV cache**. Launchers and every lever with its reason:
[`deployment/v100_qwen38/`](deployment/v100_qwen38/README.md).

| | start (int8 port, no speculation) | now |
|---|---|---|
| decode, short prompt | 32-35 tok/s | **66 tok/s** on code (3.89 tokens accepted per step), 48-51 on prose |
| decode at 210K context | — | 39 tok/s |
| 250K-token prefill | 334 s dense | **198 s**, 12/12 needles, prefix-cache repeat x10-18 |
| context on two cards | 97K tokens | **262 144 tokens** |

The numbers below were measured on this hardware, one change at a time. Where a change
has no isolated number, the mechanism is given instead.

### Speculative decoding (the main multiplier)

| Approach | Lever | Measured effect |
|---|---|---|
| Native MTP head of the model, k=3 | `--speculative-config {"method":"mtp",...,"num_speculative_tokens":3}` | with prefix cache and full graph 33.1 -> 50.5 tok/s short; k=3 is the optimum (56.9 vs 47.5 at k=1, 55.5 at k=7) |
| Speculation inside the CUDA graph | backend declares `UNIFORM_BATCH` (query_len = k+1) | without it the step ran eagerly: 277 ms vs 29.5 ms |
| k+1 must be a captured graph size | `cudagraph_capture_sizes [1,2,4,8]` | k=5 halved the speed (draft ran without a graph); valid k are 1, 3, 7 |
| Virtual batch: the k+1 positions are rows of one batch | attention kernel | MTP decode 11.3 -> 17.3 tok/s (3.3 -> 7.3 at long context) |
| Virtual batch without host copies | kernel reads the request as `r/q` | +3.7 %, output byte-identical |
| MTP head in int8 on our kernel | `FA2SM70_MTP_W8=1`, `FA2SM70_MTP_W8_NSPLIT=4` | head pass x2.74, -202 MiB per rank |
| Draft body on the tensor-core int8 path | `FA2SM70_DRAFT_TM8=1` | draft uses turbomind int8 instead of scalar gemv |
| Draft from n-grams of the context | `FA2SM70_MTP_NGRAM=1` | accepted length 3.86 vs 2.56 for MTP alone; 71.6 tok/s on repetitive text |
| Source-selection rule that does not get stuck | `FA2SM70_MTP_NGRAM_FIX=1` | n-gram searches -72 %, keeps "search" mode on code-heavy traffic |

### Linear bodies (decode is bound by reading the weights)

| Approach | Lever | Measured effect |
|---|---|---|
| Checkpoint in GPTQ int8 (RTN, group 128, asymmetric) | model format | 55 -> 32 GB: half the bytes per step; tensor error 6e-3 |
| Own int8 GEMV for M=1 | `FA2SM70_W8=1` | x2.0-2.7 over the stock kernel, 73-86 % of the read roofline |
| tm8: TurboMind GPTQ-8 on tensor cores for the M=k+1 zone | `FA2SM70_TM8=1`, `FA2SM70_TM8_ONLY=1`, `FA2SM70_TM8_NMIN=3072` | flat in M at 93 % of roofline; cost per extra draft position 4.2 -> 2.0 ms |
| Per-start autotune of tm8 variant / split-K / tile | `FA2SM70_TM8_TUNE=1` | makes tm8 win on narrow shapes too |
| Column width of the GEMV at M=4 | kernel rule | x1.33-1.53; `down_proj` 46 -> 71 % of roofline |
| Attention weights (qkv) in 12 bits | `FA2SM70_W12=1` | 0.75 byte per weight instead of 2, more precise than bf16 |
| Vocabulary projection in int8 | `FA2SM70_LMH=1` | 98 % of the read roofline; dense copy freed (-620 MB per rank) |
| Large-M bodies through reconstruct + cuBLAS | `FA2SM70_TM8_RECON_MMIN=1024`, `FA2SM70_TM8_RECON_OUT=1` | +2.8 % prefill |

### Graph and host work

| Approach | Lever | Measured effect |
|---|---|---|
| Full CUDA graph for decode | `FA2SM70_CG=1`, `cudagraph_mode=full_and_piecewise` | x1.084-1.091 |
| GDN metadata built once per step, not once per KV group (10 groups) | shared computation | 9.9 -> 6.37 ms per step |
| The graph reads the addresses of buffers that are actually filled | shared, capture-time buffers | GDN metadata 6.37 -> 1.13 ms; 42.2 tok/s |
| Shared buffers instead of copies | `FA2SM70_GDN_SHARED_BUF=1` | 2.43 -> 1.34 ms; 42.8 -> 44.2 tok/s |
| Removed work nobody read | `fused_gdn_gating` | was computed 48 times per step; removed byte-identically |
| Exact top-k/top-p without two full sorts of the 248 320 vocabulary | `FA2SM70_TOPKP_FAST=1` | +1.6-2.0 %, 0 of 10 563 live rows differ |
| No Python in the hot path | — | Python in the path cost 12.5 % and broke TP |

### Attention and KV cache (grows with context)

| Approach | Lever | Measured effect |
|---|---|---|
| int8 KV cache, per-token-per-head scale | `--kv-cache-dtype int8_per_token_head` | half the KV reads; 262 144 tokens on two cards |
| GDN state in int16, separate pools, state block x8 | `FA2SM70_GDN_I16=1`, `FA2SM70_SPLIT_POOLS=1`, `FA2SM70_MAMBA_BLK_MULT=8` | the state paid 48 of 69 KiB per token; 250K fits together with speculation |
| Hybrid decode kernel, head grouping GF=6 | kernel | optimum on three axes (row fusion 0.66x, extra splits worse) |
| Decode path chosen by length: uniform up to ~65K, prefill-style beyond | `FA2SM70_UNIFORM_MAXLEN=65000` | 13.5K: 29.9 -> 44.5 tok/s |
| Sparse decode, 15 % of blocks per row and head | `FA2SM70_SPARSE_DEC=1`, `_TOPF=0.15`, `_MINBLK=256` | x1.16; switches itself off below 16K tokens |
| 64 splits, not 160 | `FA2SM70_MAX_SPLITS=64` | 160 was tuned for dense decode and shattered sparse work |
| Sparse prefill with a per-dimension bound (Hoelder) | `FA2SM70_SPARSE_MASS=0.85`, `_GAMMA=0.5`, `_MINSQ=2048` | 250K prefill 334 -> 198 s at dense-equal quality (12/12, ladder 32/32) |

### Measured and switched off

| Tried | Why it is off |
|---|---|
| 12-bit compression of the TP exchange on NVLink | x0.79-0.81, a loss (`FA2SM70_EXCH12=0`) |
| Own all-reduce | hangs; the full graph already brought the exchange cost to 1.5 % |
| Candidate tree in the kernel | the accepted branch corrupts context: 1.835 -> 1.762 accepted per step |
| DFlash2 block draft | hurts at long context (x0.76 at 8K) and zeroes prefix-cache hits |

**How the 66 tok/s adds up.** Without speculation the full-graph decode step is 30.5 ms:
weights 15.9 (the floor), attention 3.6, exchange 2.6, GDN almost zero. Speculation
turns one step into several tokens: 3.89 accepted per step at a ~59 ms step gives 66
tok/s on code-like text. Prose accepts less, hence 48-51 tok/s there.

## Project Focus

- **V100 / SM70 first**: optimized for Tesla V100 rather than being a generic
  multi-hardware fork.
- **AWQ on Volta**: AWQ 4-bit inference paths for dense and MoE Qwen models on
  SM70.
- **V100 FlashAttention path**: `FLASH_ATTN_V100` decode and prefill backend
  for Volta GPUs, with SM70 compile-graph, guarded XQA decode, and D=256
  paged-prefix low-smem fast paths enabled by default.
- **Long-context serving**: public profiles default to 256K context where the
  model and memory budget allow it.
- **MTP serving**: Qwen3.6-class MTP speculative decoding remains available as
  an explicit opt-in path; long-context public profiles default to no MTP.
- **Image inputs by default**: SM70 `FLASH_ATTN_V100` profiles allow one image
  per prompt by default; video inputs remain opt-in.
- **Tool calling and OpenAI API compatibility**: validated with OpenAI-style
  clients such as Cherry Studio, OpenClaw, and similar tools.
- **Experimental FP8 work**: FP8 model and KV-cache paths are included for
  validation, but they are not production defaults.
- **Experimental DFlash work**: included for continued research and validation.

## Recommended Model Providers

- `tclf90/Qwen3.6-27B-AWQ`
- `tclf90/Qwen3.6-35B-A3B-AWQ`
- `tclf90/Qwen3.5-122B-A10B-AWQ` for larger 4-GPU setups

The launch examples use local paths such as `/path/to/Qwen3.6-27B-AWQ`.
Replace them with your local model path or a Hugging Face repository id.

## Hardware Target

The public commands are written for V100 Qwen serving workloads. Image inputs
are enabled by default on the SM70 `FLASH_ATTN_V100` path; video inputs are
disabled by default and should be enabled explicitly only after local memory
validation.

| Host | Notes |
| --- | --- |
| 4 x Tesla V100 32 GB | Main public reference target |
| 2 x Tesla V100 32 GB | Supported for selected 27B profiles with lower concurrency |

Typical model placement:

- `Qwen3.6-27B-AWQ`: TP1/TP2/TP4 supported; TP4 is the public reference.
- `Qwen3.6-27B-AWQ + MTP`: explicit opt-in profile for local validation, not
  the long-context public default.
- `Qwen3.6-35B-A3B-AWQ`: TP4 recommended.
- `Qwen3.5-122B-A10B-AWQ`: TP4 supported for larger deployments.

Multimodal defaults:

- Default SM70 `FLASH_ATTN_V100` serving allows `image=1`, `video=0` when
  `--limit-mm-per-prompt` is not set.
- For text-only serving, pass `--limit-mm-per-prompt '{"image":0,"video":0}'`
  or use `--language-model-only`.
- For video workloads, pass an explicit limit such as
  `--limit-mm-per-prompt '{"image":1,"video":1}'` and retune memory settings.

## Validated Stack

The public wheel path is validated on:

- OS: Ubuntu 24.04 LTS
- Python: 3.12
- CUDA toolkit: 12.8
- PyTorch: CUDA 12.8 runtime wheels
- GPU: Tesla V100 32 GB

## Quick Start

### 1. Install CUDA 12.8

Use the official NVIDIA repository on Ubuntu 24.04:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

If the machine also has another CUDA toolkit installed, force build-time and
runtime CUDA to 12.8:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
hash -r
nvcc -V
```

### 2. Create the Python environment

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda create -y -n 1cat-vllm-sm70 python=3.12
conda activate 1cat-vllm-sm70

python -m pip install --upgrade pip setuptools wheel
```

### 3. Install from Prebuilt Wheels

Prebuilt wheels are the recommended installation path for public users. Source
builds are intended for kernel development.

Download the latest wheel assets from:

```text
https://github.com/1CatAI/1Cat-vLLM/releases/latest
```

Install the wheel from the directory where you downloaded it:

```bash
python -m pip install --prefer-binary --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  ./1cat_vllm-*.whl
```

Notes:

- The `1cat_vllm` wheel already bundles the `flash_attn_v100` Python package
  and SM70 CUDA extensions.
- Runtime installation from wheels does not require the bundled `lmdeploy`
  source tree.
- Use Python 3.12 and CUDA 12.8.
- If your shell has a broken local proxy configured, unset it before
  installing:
  `env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy ...`.
- After installing from wheels, run `python -m vllm...` from a directory
  outside this source checkout, such as `cd ~` or `cd /tmp`. Running inside the
  cloned repository makes Python import the local source tree instead of the
  wheel-installed CUDA extensions.

### 4. Verify the Environment

```bash
python - <<'PY'
import torch, triton, vllm, sys
import flash_attn_v100
from flash_attn_v100 import flash_attn_v100_cuda, paged_kv_utils
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("triton", triton.__version__)
print("vllm", vllm.__version__)
print("flash_attn_v100", flash_attn_v100.__version__)
PY
```

## Recommended Launch Commands

These are the recommended public serving commands for the 27B AWQ and 35B AWQ
V100 profiles. When using prebuilt wheels, run them outside the source checkout
so Python loads the installed package and its CUDA extensions.

Use `CUDA_VISIBLE_DEVICES=0,1,2,3` only when you need to select a specific
four-card V100 set.

### Qwen3.6-27B-AWQ, TP4

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.6-27B-AWQ \
  --served-model-name qwen3.6-27b-awq \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port 8000
```

### Qwen3.6-35B-A3B-AWQ, TP4

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3.6-35B-A3B-AWQ \
  --served-model-name qwen3.6-35b-a3b-awq \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 262144 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --host 0.0.0.0 \
  --port 8000
```

## OpenAI-Compatible Request Example

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer EMPTY' \
  -d '{
    "model": "qwen3.6-27b-awq",
    "messages": [{"role": "user", "content": "用一句话回答，2+2等于几？"}],
    "temperature": 0,
    "max_completion_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

If the response is coherent and short, the API path is basically healthy.

## Experimental Features

### FP8

FP8 support is included for validation and research. It is not the stable
public default.

- FP8 model execution on V100 is experimental.
- `fp8_e5m2` KV cache can be used experimentally on V100.
- `fp8_e4m3` is not the recommended V100 option in the current path.
- Do not add `--calculate-kv-scales` unless you are specifically testing KV
  scale calculation behavior.

Example:

```bash
--kv-cache-dtype fp8_e5m2
```

### DFlash

DFlash is included as an experimental path for continued validation. Treat it
as a research feature until you have validated speed and output quality on your
own workload.

### MTP

MTP is not enabled by default in the V100 public serving profile. Long-context
decode on V100 can slow down significantly when MTP is enabled, so keep the
default no-MTP path for 128K/256K style serving unless your own workload proves
otherwise.

To explicitly test the previous automatic SM70 MTP4 profile:

```bash
export VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1
```

You can also pass an explicit `--speculative-config` when you want full control
over speculative decoding settings.

### Dense F16 Fast Path

`VLLM_SM70_ENABLE_DENSE_F16_FASTPATH=1` is intended for targeted experiments.
Keep it disabled for public MoE serving profiles unless you are explicitly
benchmarking that path.

## Source Build

Source build is supported, but it is **not recommended** for normal runtime
deployment. Install the release wheels first unless you are changing CUDA,
C++, or Triton code.

This repository includes the validated `lmdeploy` source tree under
`csrc/sm70_turbomind/lmdeploy`, which is needed by the SM70 AWQ build path.

```bash
cd /path/to/1Cat-vLLM/vllm
test -d csrc/sm70_turbomind/lmdeploy
```

Install build dependencies:

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate 1cat-vllm-sm70

python -m pip install -r requirements/build/cuda.txt
python -m pip install -r requirements/cuda.txt
python -m pip install -r requirements/common.txt
python -m pip install cmake build
```

Build wheels:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.0;8.0"
export FLASH_ATTN_V100_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=12
export NVCC_THREADS=1

rm -rf build vllm.egg-info
rm -rf .deps/*-build .deps/*-subbuild

pushd flash-attention-v100
python -m build --wheel --no-isolation --outdir ../dist-cu128-sm70
popd

python -m build --wheel --no-isolation --outdir dist-cu128-sm70
```

For editable development:

```bash
python -m pip install -e . --no-build-isolation
```

## Benchmarking Notes

- First-request warmup is slow on V100 and should not be included in
  steady-state throughput.
- Browser-side OpenAI streaming throughput includes request overhead and should
  not be compared directly with strict incremental decode TPS.
- Long-context throughput depends strongly on TP, `max_num_seqs`,
  `max_num_batched_tokens`, prompt shape, and attention backend.
- If you publish a baseline, include the full launch command, GPU model,
  driver, CUDA runtime, model checkpoint, sampling parameters, prompt length,
  and decode length.

## WeChat Community

**群聊：** 1Cat-vLLM 开源交流群

请使用微信扫描下方二维码加入群组：

![1Cat-vLLM 微信交流群二维码](docs/assets/wechat-group-qr-4.png)

> 提示：微信群二维码通常 7 天内有效。若扫描失败或提示过期，请重新打开本页查看最新图片，或关注仓库更新。

## Repository Notes

- Upstream project: [vLLM](https://github.com/vllm-project/vllm)
- This fork focuses on SM70 AWQ support, V100-oriented attention/runtime
  tuning, and experimental FP8/MTP/DFlash validation paths.
- Prebuilt wheels are the public installation path.
- Source builds are for development and kernel work.

## Acknowledgements

- [vLLM](https://github.com/vllm-project/vllm)
- [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy)
- [flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)
- [marlin_v100](https://github.com/zhinianqin/marlin_v100)

## License

This repository follows the upstream vLLM license model. See [LICENSE](LICENSE).
