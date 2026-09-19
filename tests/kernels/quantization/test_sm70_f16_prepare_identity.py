# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the shipped `sm70_f16_*` dense fp16 operator (sm_70 / V100).

Written as a FAILING test first, against the SHIPPED binary, for задача #137.
Source of the defects: /mnt/d1/alex/reports/EV_shapes_production.md §5.

Three things are asserted here, and each one used to be violated silently:

  1. IDENTITY.  `sm70_f16_prepare` memoises the converted weight in a process-global
     table keyed on the raw `TensorImpl*` of the source weight -- an address the table
     does not own.  Free a weight and the very next tensor of the same shape lands on
     that address and gets the PREVIOUS weight's conversion back.  Measured symptom:
     relL2 == sqrt(2) with identical k_ld and shapes, cured by re-preparing a clone.
     `test_prepare_does_not_hand_back_a_dead_weights_conversion` triggers it on
     purpose and deterministically (8/8 on the shipped .so, at three shapes): free the
     weight's TensorImpl LAST, then allocate the next weight immediately, so malloc
     hands the slot straight back.

  2. SHAPE.  The packed sm70 HMMA.884 B-operand layout groups output rows in 32s
     (`PackingImpl<HMMA_884, OPERAND_B, 1, kRowMajor>::apply({n,k}) == {n/32, k*32}`),
     so N % 32 != 0 silently DROPS the tail rows.  The check for it lived in the Python
     caller (`process_weights_after_loading`), not in the operator, so a direct call
     with N=48 (that is `linear_attn.in_proj_ba` at TP=2) was accepted and returned a
     wrong transform (measured relL2 0.59...1.26).

  3. SIGNALLING.  `sm70_f16_gemm_out` took `k_ld` on trust.  For this converter the
     packed leading dimension is always 32*K, so a mismatched (tm_weight, k_ld) pair is
     detectable and must be refused instead of computing garbage.

Run against an arbitrary build (this is how the A/B against the shipped .so is done):

    FA2SM70_C_SO=/path/to/_C.abi3.so CUDA_VISIBLE_DEVICES=1 \
        python tests/kernels/quantization/test_sm70_f16_prepare_identity.py

Without `FA2SM70_C_SO` it uses the installed `vllm._C`.  The driver runs every case in
its OWN process: the value-level reproducer below depends on the heap layout, and a
case that ran before it can hide the defect (observed).  `FA2SM70_TEST_ORDER` picks the
cases and their order -- the fix has to hold in every order.
"""

import gc
import os
import subprocess
import sys

import pytest
import torch

K = 5120  # боевая форма self_attn.qkv_proj: K=5120
N = 14336  # N=14336
MS = (1, 8, 32, 128, 512, 2048, 8192)
REL_TOL = 1e-2


def _load_ops():
    so = os.environ.get("FA2SM70_C_SO")
    if so:
        torch.ops.load_library(so)
    else:
        import vllm._C  # noqa: F401
    return torch.ops._C


def _need_sm70():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("sm_70 required")


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref).norm() / ref.norm()).item()


# --------------------------------------------------------------------------- 1
@pytest.mark.parametrize("n,k", [(32, 16), (64, 512), (N, K)])
def test_prepare_does_not_hand_back_a_dead_weights_conversion(n, k):
    """ДЕФЕКТ 1, прямой фальсификатор: не «похоже на мусор», а «чей это вес».

    Освобождаем TensorImpl веса ПОСЛЕДНИМ и тут же выделяем следующий -- malloc отдаёт
    тот же слот, и таблица, ключом которой этот адрес и является, отвечает за чужой
    вес.  `tmw1` держим живым, поэтому его буфер не может быть переиспользован честно:
    совпадение data_ptr означает ровно одно -- вернули ту же самую запись.
    """
    _need_sm70()
    ops = _load_ops()

    # `.mul_` and not `* 0.05`: the multiply would allocate an intermediate TensorImpl
    # and the slot the test is aiming at would go to it instead of to w2.
    w1 = torch.randn(n, k, device="cuda", dtype=torch.float16).mul_(0.05)
    p1 = ops.sm70_f16_prepare(w1)
    tmw1 = p1[0]
    ptr1 = tmw1.data_ptr()
    del p1
    gc.collect()
    del w1

    w2 = torch.randn(n, k, device="cuda", dtype=torch.float16).mul_(0.05)
    p2 = ops.sm70_f16_prepare(w2)
    tmw2, kld2 = p2[0], int(p2[1][0].item())

    assert tmw2.data_ptr() != ptr1, (
        f"sm70_f16_prepare returned the conversion of an already freed weight (same "
        f"buffer {hex(ptr1)}) for a new {n}x{k} weight: the memo table keyed on a raw "
        f"TensorImpl* it does not own, and malloc reissued the address."
    )
    assert not torch.equal(tmw2, tmw1), (
        "bit-identical transform for two independent random weights -- that is a "
        "stale memo hit, not a coincidence."
    )

    x = torch.randn(32, k, device="cuda", dtype=torch.float16) * 0.05
    ref = x.float() @ w2.float().t()
    out = torch.empty((32, n), dtype=torch.float16, device="cuda")
    ops.sm70_f16_gemm_out(out, x, tmw2, kld2, False)
    assert _rel_l2(out, ref) < REL_TOL
    assert kld2 == 32 * k, "packed leading dim must be 32*K for the sm70 884 converter"


# --------------------------------------------------------------------------- 2
def test_repro_block_a_prepare_after_other_work():
    """ДЕФЕКТ 1 в значениях: воспроизводитель из отчёта, блок A (18 отказов из 21).

    Обвязка дословно как в `raw/EV_shapes/repro_sm70_f16_wrong.py`: раскладка кучи
    здесь и есть спуск, поэтому ни лишних вспомогательных кадров, ни gc между шагами.
    """
    _need_sm70()
    ops = _load_ops()
    torch.manual_seed(0)

    bad = []
    for rep in range(3):
        for m in MS:
            x = torch.randn(m, K, device="cuda", dtype=torch.float16) * 0.05
            wl = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05
            ref = x.float() @ wl.float().t()  # работа между выделением и подготовкой
            y = torch.nn.functional.linear(x, wl)
            _ = ((y.float() - ref).norm() / ref.norm()).item()
            del y
            p = ops.sm70_f16_prepare(wl)
            tmw, kld = p[0], int(p[1][0].item())
            o = torch.empty((m, tmw.shape[0]), dtype=torch.float16, device="cuda")
            ops.sm70_f16_gemm_out(o, x, tmw, kld, False)
            r = ((o.float() - ref).norm() / ref.norm()).item()
            if r >= REL_TOL:
                bad.append((rep, m, round(r, 4)))
            del x, wl, ref, o, p, tmw
            torch.cuda.empty_cache()
    assert not bad, f"sm70_f16 answered with an unrelated matrix at {bad}"


# --------------------------------------------------------------------------- 3
def test_repro_block_b_prepare_first_after_accumulated_state():
    """ДЕФЕКТ 1, блок B: порядок, безопасный в ЧИСТОМ процессе, но после блока A."""
    _need_sm70()
    ops = _load_ops()
    torch.manual_seed(1)

    for _rep in range(3):  # накопить состояние ровно тем же телом, что и блок A
        for m in MS:
            x = torch.randn(m, K, device="cuda", dtype=torch.float16) * 0.05
            w = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05
            ref = x.float() @ w.float().t()
            y = torch.nn.functional.linear(x, w)
            _ = ((y.float() - ref).norm() / ref.norm()).item()
            del y
            p = ops.sm70_f16_prepare(w)
            tmw, kld = p[0], int(p[1][0].item())
            o = torch.empty((m, tmw.shape[0]), dtype=torch.float16, device="cuda")
            ops.sm70_f16_gemm_out(o, x, tmw, kld, False)
            del x, w, ref, o, p, tmw
            torch.cuda.empty_cache()

    wl = torch.randn(N, K, device="cuda", dtype=torch.float16) * 0.05
    p = ops.sm70_f16_prepare(wl)
    tmw, kld = p[0], int(p[1][0].item())
    bad = []
    for m in MS:
        x = torch.randn(m, K, device="cuda", dtype=torch.float16) * 0.05
        ref = x.float() @ wl.float().t()
        o = torch.empty((m, tmw.shape[0]), dtype=torch.float16, device="cuda")
        ops.sm70_f16_gemm_out(o, x, tmw, kld, False)
        r = ((o.float() - ref).norm() / ref.norm()).item()
        if r >= REL_TOL:
            bad.append((m, round(r, 4)))
        del x, ref, o
    assert not bad, f"sm70_f16 answered with an unrelated matrix at {bad}"


# --------------------------------------------------------------------------- 4
def test_prepare_rejects_rows_not_multiple_of_32():
    """ДЕФЕКТ 2: N=48 (in_proj_ba при TP=2) обязан быть ОТВЕРГНУТ оператором."""
    _need_sm70()
    ops = _load_ops()
    w = torch.randn(48, K, device="cuda", dtype=torch.float16) * 0.05
    with pytest.raises(RuntimeError, match="32"):
        ops.sm70_f16_prepare(w)


def test_prepare_rejects_cols_not_multiple_of_16():
    _need_sm70()
    ops = _load_ops()
    w = torch.randn(64, 5000, device="cuda", dtype=torch.float16) * 0.05
    with pytest.raises(RuntimeError, match="16"):
        ops.sm70_f16_prepare(w)


# --------------------------------------------------------------------------- 5
def test_gemm_out_rejects_a_mismatched_k_ld():
    """ТРЕТЬЕ: gemm_out обязан отказывать, а не считать по чужому k_ld."""
    _need_sm70()
    ops = _load_ops()
    w = torch.randn(64, K, device="cuda", dtype=torch.float16) * 0.05
    p = ops.sm70_f16_prepare(w)
    tmw, kld = p[0], int(p[1][0].item())
    assert kld == 32 * K
    x = torch.randn(8, K, device="cuda", dtype=torch.float16) * 0.05
    out = torch.empty((8, 64), dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        ops.sm70_f16_gemm_out(out, x, tmw, kld // 2, False)


def test_gemm_out_rejects_weight_rows_not_multiple_of_32():
    _need_sm70()
    ops = _load_ops()
    tmw = torch.zeros(48, K, dtype=torch.float16, device="cuda")
    x = torch.randn(8, K, device="cuda", dtype=torch.float16) * 0.05
    out = torch.empty((8, 48), dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="32"):
        ops.sm70_f16_gemm_out(out, x, tmw, 32 * K, False)


# --------------------------------------------------------------------------- 6
def test_live_weights_still_hit_the_memo_table():
    """Правка не должна убить сам кэш: живой вес обязан отдаваться из таблицы."""
    _need_sm70()
    ops = _load_ops()
    w = torch.randn(64, K, device="cuda", dtype=torch.float16) * 0.05
    a = ops.sm70_f16_prepare(w)
    b = ops.sm70_f16_prepare(w)
    assert a[0].data_ptr() == b[0].data_ptr(), "memo table stopped working"


def test_prepare_survives_a_weight_that_moved_storage():
    """`param.data = other` оставляет TensorImpl, но меняет хранилище."""
    _need_sm70()
    ops = _load_ops()
    w = torch.randn(64, K, device="cuda", dtype=torch.float16) * 0.05
    first = ops.sm70_f16_prepare(w)[0].clone()
    w.data = (torch.randn(64, K, device="cuda", dtype=torch.float16) * 0.05).data
    second = ops.sm70_f16_prepare(w)
    assert not torch.equal(second[0], first), (
        "prepare returned the OLD storage's transform after the weight was re-pointed"
    )
    x = torch.randn(8, K, device="cuda", dtype=torch.float16) * 0.05
    ref = x.float() @ w.float().t()
    out = torch.empty((8, 64), dtype=torch.float16, device="cuda")
    ops.sm70_f16_gemm_out(out, x, second[0], int(second[1][0].item()), False)
    assert _rel_l2(out, ref) < REL_TOL


_CASES = {
    "i": ("identity 32x16", lambda: test_prepare_does_not_hand_back_a_dead_weights_conversion(32, 16)),
    "I": ("identity 14336x5120", lambda: test_prepare_does_not_hand_back_a_dead_weights_conversion(N, K)),
    "a": ("repro block A", test_repro_block_a_prepare_after_other_work),
    "b": ("repro block B", test_repro_block_b_prepare_first_after_accumulated_state),
    "n": ("N%32 refused by prepare", test_prepare_rejects_rows_not_multiple_of_32),
    "k": ("K%16 refused by prepare", test_prepare_rejects_cols_not_multiple_of_16),
    "l": ("bad k_ld refused by gemm_out", test_gemm_out_rejects_a_mismatched_k_ld),
    "g": ("N%32 refused by gemm_out", test_gemm_out_rejects_weight_rows_not_multiple_of_32),
    "m": ("memo table still works", test_live_weights_still_hit_the_memo_table),
    "s": ("weight moved storage", test_prepare_survives_a_weight_that_moved_storage),
}
_DEFAULT_ORDER = "iIabnklgms"


def _main() -> int:
    order = os.environ.get("FA2SM70_TEST_ORDER") or _DEFAULT_ORDER
    one = os.environ.get("FA2SM70_TEST_ONE")
    if one:
        name, fn = _CASES[one]
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            head = str(exc).strip().splitlines()[0][:200]
            print(f"  FAIL  {name}: {type(exc).__name__}: {head}")
            return 1
        print(f"  PASS  {name}")
        return 0

    failed = 0
    for ch in order:
        env = dict(os.environ, FA2SM70_TEST_ONE=ch)
        rc = subprocess.run([sys.executable, __file__], env=env).returncode
        failed += rc != 0
    print(f"итог: отказов {failed} из {len(order)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
