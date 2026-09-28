# Validated deployment: Qwen3.8-27B int8 on 2x Tesla V100 (SM70)

Production configuration measured on two V100-SXM2-32GB, tensor parallel 2,
NVLink, CUDA 12.8. Every lever below is defended by a measurement, and every
quality claim by a gate that can be re-run from `tools/` of the kernel repo
(xormal/V100-SM_70-Flash-Attn-2-v1).

## Levers

| lever | value | why |
|---|---|---|
| `FA2SM70_KV_BYTES` | 10630627328 | 262 144 tokens of int8 KV across both cards |
| `FA2SM70_MAXBATCH` | 2048 | prefill chunk; 1024 costs 8% of prefill |
| `FA2SM70_SPARSE_MASS` | 0.85 | mass kept per tile by the prefill selector |
| `FA2SM70_SPARSE_MINSQ` | 2048 | moves with the chunk: below it the last partial chunk stays dense |
| `FA2SM70_SPARSE_DEC` / `_TOPF` | 1 / 0.15 | per-row, per-head block selection in decode |
| `FA2SM70_SPARSE_DEC_MINBLK` | 256 | below 16 384 tokens the selection goes dense by itself |
| `FA2SM70_MAX_SPLITS` | 64 | 160 was tuned for dense decode; with 15% of blocks it shatters the work |
| `FA2SM70_LMH` | 1 | int8 vocabulary projection, dense fp16 weight freed (-620 MB per rank) |
| `FA2SM70_MTP_W8_NSPLIT` | 4 | splits the draft head body by columns; the foreign dequant buffer drops 170 -> 42 MiB |
| `FA2SM70_W8_OOM_RETRY` | 1 | an allocator shortage returns the cache and retries instead of killing the worker |
| `FA2SM70_MTP_NGRAM` | 1 | context n-grams as the draft source; neutral on prose, large on copying |
| `FA2SM70_CG` | 1 | full CUDA graph for decode |
| `FA2SM70_TOPKP_FAST` | 1 | top-k/top-p without two full sorts of the 248 320 vocabulary per step; same tie order as the reference, 0 of 10 563 live rows differ |
| `FA2SM70_MTP_NGRAM_FIX` | 1 | per-source running acceptance with search/explore modes; the old rule got stuck on one source |
| `FA2SM70_TM8_RECON_OUT` | 1 | the reconstructed body writes `torch.mm` straight into the output slice (`out=`), no extra copy |
| `BOEVAYA_SET` | `abl` / `cyber` | which checkpoint the launcher serves; each has its own w12 store and compile cache, so switching back is one variable |

## Measured on production hardware

| gate | result |
|---|---|
| 12 needles at 250K, real text | 12/12 |
| ladder 8 / 16 / 24 / 32 needles at 100K | 32/32 |
| block-boundary probe, 5 arms x 10 runs | 400/400 needles, 0 losses |
| multi-needle 4K..200K | 8/8 at every length |
| 8 concurrent image requests | 8/8, engine alive |
| prefix cache at 250K | x18.5, answers byte-identical on repeat |
| 250K prefill | 192.7-218.8 s (the spread is thermal, see below) |
| decode, slope instrument | 48-51 tok/s short, ~39 at 210K |

## Two things worth knowing before you tune

**Measure decode by slope.** Wall time around one request drowns in prefix-cache
work at long context (15 vs 47 tok/s at 250K), and SSE streaming halves the
number even on a 300-token prompt because the API server becomes the bottleneck.
Use the same prompt twice with different `max_tokens` and divide the differences.

**The cards throttle.** Under a 250K prefill a V100 goes from 48 to 82 C in 90
seconds and the SM clock falls from 1500 to 1147-1290 MHz against a 1530 ceiling,
with `hw_thermal_slowdown` active. The prefill is under 200 s on a cold card and
216-219 s on a hot one. Quote thresholds together with the thermal state.
