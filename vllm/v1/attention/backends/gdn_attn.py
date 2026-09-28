# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend for GatedDeltaNet attention."""

import json
import os
import time
from dataclasses import dataclass
from typing import Literal

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

logger = init_logger(__name__)


# [FA2/SM70 25.08] ФАЗОМЕР СТРОИТЕЛЯ (FA2SM70_GDN_PHASE=N). Замер показал: этот строитель стоит
# 9.9 мс НА ШАГ при спекуляции -- больше, чем всё остальное вне прохода. Надо назвать УЧАСТОК.
import os as _os
import time as _time

_ГФ = {}
_ГС = [0]
# [ПРИБОР НЕ ДОЛЖЕН СТОИТЬ, КОГДА ВЫКЛЮЧЕН, 31.08]
# `_гф` зовётся ВОСЕМЬ раз в каждом `build`, а `build` -- на каждую группу KV каждого шага
# (у нас их десять-двенадцать): около девяноста вызовов за шаг. В каждом стоял
# `int(os.environ.get(...))` -- обращение к словарю окружения плюс разбор строки, и всё это
# при ВЫКЛЮЧЕННОМ приборе. Рычаг читаем один раз при импорте.
_ГФ_ВКЛ = int(_os.environ.get("FA2SM70_GDN_PHASE", "0"))
# [ПОЛОЖИТЕЛЬНЫЙ КОНТРОЛЬ ХОЗЯЙСКОГО ПУТИ -- 08.09]
# Прежде чем переписывать построение метаданных на нативное ядро (пункт 5 цели, ~2.2 мс),
# надо ответить на вопрос, который дороже самой правки: ЛЕЖИТ ЛИ хозяйская работа на
# критическом пути? Если поток хозяина всё равно ждёт карту, снятие питона даст НОЛЬ.
# Ответ даёт положительный контроль: ДОБАВИТЬ хозяйской работы и посмотреть на шаг.
# Ветвь выбирается по файлу (чередование внутри одного экземпляра, §212), задержка -- в
# микросекундах. Пустой путь = ветка мертва.
_ЗАДЕРЖКА_ФАЙЛ = _os.environ.get("FA2SM70_HOST_DELAY_FILE", "")
_ЗАДЕРЖКА_МКС = int(_os.environ.get("FA2SM70_HOST_DELAY_US", "0"))
_ЗАД = {"вкл": False, "счёт": 0}


def _задержка_хозяина():
    if not _ЗАДЕРЖКА_ФАЙЛ or _ЗАДЕРЖКА_МКС <= 0:
        return
    _ЗАД["счёт"] += 1
    if _ЗАД["счёт"] % 64 == 1:
        # Содержимое файла -- ЧИСЛО микросекунд (0 = ветка мертва). Так одна сборка даёт
        # весь свип величин, а не одну; чередование с нулём снимает дрейф стенда.
        try:
            with open(_ЗАДЕРЖКА_ФАЙЛ) as ф:
                _т = ф.read(16).strip()
            _ЗАД["мкс"] = int(_т) if _т.lstrip("-").isdigit() else (
                _ЗАДЕРЖКА_МКС if _т[:1] == "1" else 0
            )
        except (OSError, ValueError):
            _ЗАД["мкс"] = 0
    _м = _ЗАД.get("мкс", 0)
    if _м > 0:
        _к = _time.perf_counter() + _м * 1e-6
        while _time.perf_counter() < _к:
            pass


def _гф(имя, т0):
    if not _ГФ_ВКЛ:
        return
    _ГФ[имя] = _ГФ.get(имя, 0.0) + (_time.perf_counter() - т0) * 1e3


def _гпечать():
    н = _ГФ_ВКЛ
    if not н:
        return
    _ГС[0] += 1
    if _ГС[0] % н == 0:
        c = _ГС[0]
        стр = "  ".join(f"{и}={в / c:.2f}" for и, в in sorted(_ГФ.items(), key=lambda x: -x[1]))
        print(f"[gdn ФАЗЫ СТРОИТЕЛЯ, мс/вызов, {c}] {стр}", flush=True)


_SM70_GDN_STATE_TABLE_DUMP_COUNTS: dict[int, int] = {}

GDN_SPEC_METADATA_TENSORS = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
_GDN_SPEC_METADATA_TENSOR_REGISTRY: dict[str, GDN_SPEC_METADATA_TENSORS] = {}


@dataclass
class _GDNDdTreeFastCommonBuffers:
    spec_sequence_masks: torch.Tensor
    spec_token_indx: torch.Tensor
    non_spec_token_indx: torch.Tensor
    spec_query_start_loc: torch.Tensor
    num_accepted_tokens: torch.Tensor
    spec_state_slot_selectors: torch.Tensor
    initialized_key: tuple[int, int, int] | None = None
    updated_epoch: int | None = None
    token_index_initialized_size: int = 0


_GDN_DDTREE_FAST_COMMON_BUFFERS: dict[
    tuple[str, int | None, int, int], _GDNDdTreeFastCommonBuffers
] = {}


def _dflash_ddtree_gdn_shared_common_enabled() -> bool:
    return os.getenv(
        "VLLM_DFLASH_DDTREE_GDN_SHARED_COMMON", "1"
    ).strip().lower() not in ("0", "false", "no", "off", "")


def _get_ddtree_gdn_fast_common_buffers(
    device: torch.device,
    decode_cudagraph_max_bs: int,
    width: int,
) -> _GDNDdTreeFastCommonBuffers:
    key = (device.type, device.index, decode_cudagraph_max_bs, width)
    buffers = _GDN_DDTREE_FAST_COMMON_BUFFERS.get(key)
    if buffers is not None:
        return buffers
    spec_sequence_masks = torch.empty(
        (decode_cudagraph_max_bs,),
        dtype=torch.bool,
        device=device,
    )
    spec_token_indx = torch.empty(
        (decode_cudagraph_max_bs * width,),
        dtype=torch.int32,
        device=device,
    )
    non_spec_token_indx = torch.empty(
        (decode_cudagraph_max_bs * width,),
        dtype=torch.int32,
        device=device,
    )
    spec_query_start_loc = torch.empty(
        (decode_cudagraph_max_bs + 1,),
        dtype=torch.int32,
        device=device,
    )
    num_accepted_tokens = torch.empty(
        (decode_cudagraph_max_bs,),
        dtype=torch.int32,
        device=device,
    )
    spec_state_slot_selectors = torch.empty(
        (decode_cudagraph_max_bs,),
        dtype=torch.int32,
        device=device,
    )
    buffers = _GDNDdTreeFastCommonBuffers(
        spec_sequence_masks=spec_sequence_masks,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=non_spec_token_indx,
        spec_query_start_loc=spec_query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        spec_state_slot_selectors=spec_state_slot_selectors,
    )
    _GDN_DDTREE_FAST_COMMON_BUFFERS[key] = buffers
    return buffers


def _ddtree_trace_path() -> str | None:
    return os.getenv("VLLM_DFLASH_DDTREE_TRACE_JSONL")


def _dflash_ddtree_metadata_profile_enabled() -> bool:
    return os.getenv("VLLM_DFLASH_DDTREE_METADATA_PROFILE", "0") == "1"


def _dflash_ddtree_gdn_fast_build_enabled() -> bool:
    return (
        os.getenv("VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD", "0") == "1"
        and os.getenv("VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD", "0") != "1"
    )


def _dflash_ddtree_gdn_fast_build_cache_enabled() -> bool:
    return (
        os.getenv("VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD_CACHE", "0") == "1"
        and os.getenv("VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD_CACHE", "0") != "1"
    )


def _dflash_ddtree_gdn_fast_build_triton_enabled() -> bool:
    return os.getenv(
        "VLLM_DFLASH_DDTREE_GDN_FAST_BUILD_TRITON", "1"
    ).strip().lower() not in ("0", "false", "no", "off", "")


@triton.jit
def _ddtree_gdn_fast_metadata_kernel(
    state_src,
    spec_state,
    spec_sequence_masks,
    spec_query_start_loc,
    num_accepted_out,
    selector_out,
    num_accepted_src,
    selector_src,
    width: tl.constexpr,
    batch_size: tl.constexpr,
    query_len: tl.constexpr,
    tail_initialized: tl.constexpr,
    update_common: tl.constexpr,
    common_tail_initialized: tl.constexpr,
    state_block: tl.constexpr,
):
    state_offsets = tl.arange(0, state_block)
    state_mask = state_offsets < width
    tl.store(
        spec_state + state_offsets,
        tl.load(state_src + state_offsets, mask=state_mask, other=-1),
        mask=state_mask,
    )
    if update_common:
        tl.store(spec_sequence_masks, True)
        tl.store(spec_query_start_loc, 0)
        tl.store(num_accepted_out, tl.load(num_accepted_src))
        tl.store(selector_out, tl.load(selector_src))

    if not tail_initialized:
        total_state = batch_size * width
        tail_offsets = tl.arange(0, state_block)
        tail_mask = tail_offsets < (total_state - width)
        tl.store(spec_state + width + tail_offsets, -1, mask=tail_mask)

    if update_common and not common_tail_initialized:
        row_offsets = tl.arange(0, state_block)
        row_tail_mask = row_offsets < (batch_size - 1)
        tl.store(spec_sequence_masks + 1 + row_offsets, False, mask=row_tail_mask)
        tl.store(num_accepted_out + 1 + row_offsets, 1, mask=row_tail_mask)
        tl.store(selector_out + 1 + row_offsets, 1, mask=row_tail_mask)

        q_tail_mask = row_offsets < batch_size
        tl.store(spec_query_start_loc + 1 + row_offsets, query_len, mask=q_tail_mask)


def _trace_tensor(value: torch.Tensor | None) -> object:
    if value is None:
        return None
    return value.detach().cpu().tolist()


def _write_ddtree_trace_event(event: str, payload: dict[str, object]) -> None:
    trace_path = _ddtree_trace_path()
    if not trace_path:
        return
    record = {
        "event": event,
        "pid": os.getpid(),
        **payload,
    }
    try:
        with open(trace_path, "a", encoding="utf-8") as trace_file:
            json.dump(record, trace_file, ensure_ascii=True, sort_keys=True)
            trace_file.write("\n")
    except OSError:
        logger.exception("Failed to write DDTree trace event to %s", trace_path)


def _sm70_flashqla_original_prefill_enabled() -> bool:
    raw = os.getenv("VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL")
    if raw is None:
        raw = os.getenv("FLASH_QLA_SM70_USE_ORIGINAL_TILELANG")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _parse_sm70_int_ranges(raw_ranges: str | None) -> set[int] | None:
    if not raw_ranges:
        return None
    values: set[int] = set()
    for raw_part in raw_ranges.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    return values


def _dump_dflash_state_table(payload: dict[str, object]) -> str:
    dump_path = f"/tmp/dflash_state_table_pid{os.getpid()}.pt"
    torch.save(payload, dump_path)
    return dump_path


def _dump_sm70_gdn_state_table(
    payload: dict[str, object],
    seq_lens: torch.Tensor,
    num_prefills: int,
    num_decodes: int,
) -> str | None:
    dump_dir = os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_DIR")
    if not dump_dir:
        return None

    seq_lens_cpu = seq_lens.detach().cpu()
    max_seq_len = int(seq_lens_cpu.max().item()) if seq_lens_cpu.numel() else 0
    target_seqs = _parse_sm70_int_ranges(
        os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_SEQS")
    )
    if target_seqs is not None and max_seq_len not in target_seqs:
        return None
    start_seq = int(os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_START_SEQ", "0"))
    end_seq = int(os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_END_SEQ", "0"))
    if start_seq and max_seq_len < start_seq:
        return None
    if end_seq and max_seq_len > end_seq:
        return None

    pid = os.getpid()
    count = _SM70_GDN_STATE_TABLE_DUMP_COUNTS.get(pid, 0)
    max_dumps = int(os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_MAX_DUMPS", "32"))
    if count >= max_dumps:
        return None
    _SM70_GDN_STATE_TABLE_DUMP_COUNTS[pid] = count + 1

    os.makedirs(dump_dir, exist_ok=True)
    dump_path = os.path.join(
        dump_dir,
        "gdn_state_table"
        f"_pid{pid}"
        f"_dump{count:04d}"
        f"_seq{max_seq_len}"
        f"_p{num_prefills}"
        f"_d{num_decodes}.pt",
    )
    torch.save({**payload, "seq_lens_cpu_snapshot": seq_lens_cpu}, dump_path)
    return dump_path


class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]:
        return GDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    spec_state_slot_selectors: torch.Tensor | None = None  # shape: [batch,]
    ddtree_parent_ids: torch.Tensor | None = None  # shape: [batch, tree_slots]
    ddtree_num_tree_tokens_cpu: torch.Tensor | None = None  # shape: [batch,]

    # Pre-computed FLA chunk metadata (avoids GPU->CPU sync in prepare_chunk_indices)
    chunk_indices: torch.Tensor | None = None
    chunk_offsets: torch.Tensor | None = None

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None

    # --- РЕЖИМ 'all' (префикс-кэш ДЛЯ РЕКУРРЕНТНЫХ СЛОЁВ), задача 194 ---------------------
    # В режимах 'none'/'align' на запрос приходится ОДНО состояние: индекс одномерный, и
    # восстановить состояние в середине последовательности нечем -- поэтому со спекуляцией
    # 'align' портит выход (проверено гейтом тождественности 19.08, research/23).
    # В режиме 'all' состояние сохраняется НА КАЖДОЙ ГРАНИЦЕ БЛОКА, поэтому:
    #   * non_spec_state_indices_tensor становится ДВУМЕРНЫМ (запрос x блоки);
    #   * block_idx_* -- указатели В ЭТУ таблицу: откуда взять начальное состояние
    #     (последний ПОСЧИТАННЫЙ токен) и куда положить промежуточные и финальное.
    # Механизм и имена -- ровно как у Mamba2 (mamba_attn.py), чтобы два тела не разошлись.
    # ВСЕ ЧЕТЫРЕ -- ПО НЕ-СПЕКУЛЯТИВНЫМ ЗАПРОСАМ ЦЕЛИКОМ (сперва декоды, затем префиллы), а
    # не по одним префиллам, как у Mamba2. Причина в устройстве GDN: когда в батче есть хоть
    # один префилл, чанковый скан обрабатывает ВЕСЬ не-спекулятивный батч разом, и декодные
    # запросы идут тем же вызовом. Срез «только префиллы» дал бы им чужие блоки -- отказа не
    # будет, будет тихая порча.
    # [FA2/SM70 25.08] Длины нужны, чтобы пересобрать индексы состояний под таблицу блоков ДРУГОЙ
    # группы (см. update_block_table): у Mamba2 они в метаданных есть, у GDN их не было.
    seq_lens_для_обновления: torch.Tensor | None = None
    block_idx_last_computed_token: torch.Tensor | None = None
    block_idx_last_scheduled_token: torch.Tensor | None = None
    block_idx_first_scheduled_token: torch.Tensor | None = None
    num_computed_tokens_ns: torch.Tensor | None = None
    mamba_block_size: int = 0
    # [СВЁРТКА У ГРАНИЦЫ, 04.09] Колонки полной таблицы блоков: откуда спекулятивная свёртка
    # берёт состояние и куда его кладёт. Совпадают всюду, кроме шага, где опора сдвинулась.
    spec_conv_блоки: torch.Tensor | None = None
    spec_conv_чт: torch.Tensor | None = None
    spec_conv_зап: torch.Tensor | None = None
    # [ДЕРЕВО] nacc со сдвигом +W для ЧТЕНИЯ SSM при принятой ветви B (свёртке -- нельзя).
    num_accepted_ssm: torch.Tensor | None = None
    num_accepted_conv: torch.Tensor | None = None
    # Копии на стороне процессора -- ТОЛЬКО для префилла: чтобы разрезать вызов скана по
    # границам блоков, нужны сами числа, а не тензоры на карте. Стоят одной синхронизации на
    # шаг префилла (десятки миллисекунд работы) и НЕ трогаются в декоде, где синхронизация
    # ломала бы полный граф.
    num_computed_tokens_ns_cpu: torch.Tensor | None = None
    non_spec_query_start_loc_cpu: torch.Tensor | None = None


@dataclass
class GDNSpecDecodeStateContract:
    spec_state_indices_tensor: torch.Tensor
    non_spec_state_indices_tensor: torch.Tensor | None
    num_accepted_tokens: torch.Tensor
    spec_state_slot_selectors: torch.Tensor


def _empty_gdn_spec_metadata_tensors(
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_bool = torch.empty(0, dtype=torch.bool, device=device)
    return (
        empty_i32,
        empty_i32,
        empty_i32,
        empty_i32,
        empty_i32,
        empty_i32,
        empty_bool,
        empty_i32,
        empty_i32,
    )


def gdn_spec_metadata_tensors(
    attn_metadata: GDNAttentionMetadata | None,
    device: torch.device,
) -> GDN_SPEC_METADATA_TENSORS:
    """Return graph-visible active-MTP metadata tensors for Qwen GDN ops."""
    if attn_metadata is None:
        return _empty_gdn_spec_metadata_tensors(device)

    empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
    empty_bool = torch.empty(0, dtype=torch.bool, device=device)

    def _or_empty_i32(tensor: torch.Tensor | None) -> torch.Tensor:
        return tensor if tensor is not None else empty_i32

    return (
        _or_empty_i32(attn_metadata.non_spec_query_start_loc),
        _or_empty_i32(attn_metadata.non_spec_state_indices_tensor),
        _or_empty_i32(attn_metadata.spec_query_start_loc),
        _or_empty_i32(attn_metadata.spec_state_indices_tensor),
        _or_empty_i32(attn_metadata.spec_token_indx),
        _or_empty_i32(attn_metadata.non_spec_token_indx),
        (
            attn_metadata.spec_sequence_masks
            if attn_metadata.spec_sequence_masks is not None
            else empty_bool
        ),
        _or_empty_i32(attn_metadata.num_accepted_tokens),
        _or_empty_i32(
            attn_metadata.spec_state_slot_selectors
            if attn_metadata.spec_state_slot_selectors is not None
            else attn_metadata.num_accepted_tokens
        ),
    )


def register_gdn_spec_metadata_tensors(
    layer_names: list[str],
    tensors: GDN_SPEC_METADATA_TENSORS,
) -> None:
    for layer_name in layer_names:
        _GDN_SPEC_METADATA_TENSOR_REGISTRY[layer_name] = tensors


def get_registered_gdn_spec_metadata_tensors(
    layer_name: str,
    device: torch.device,
) -> GDN_SPEC_METADATA_TENSORS:
    tensors = _GDN_SPEC_METADATA_TENSOR_REGISTRY.get(layer_name)
    if tensors is None:
        return _empty_gdn_spec_metadata_tensors(device)
    if tensors[0].device != device:
        return _empty_gdn_spec_metadata_tensors(device)
    return tensors


def gather_gdn_state_block_ids(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    width: int,
) -> torch.Tensor:
    current_block_idx = torch.clamp((seq_lens - 1) // block_size, min=0)
    offsets = torch.arange(width, device=block_table.device, dtype=torch.long)
    gather_indices = current_block_idx.to(torch.long).unsqueeze(1) + offsets
    gather_indices = torch.clamp(gather_indices, max=block_table.shape[1] - 1)
    return torch.gather(block_table, 1, gather_indices)


def select_gdn_state_block_ids(
    block_table: torch.Tensor,
    accepted_tokens: torch.Tensor | None,
    num_spec: int,
) -> torch.Tensor:
    if envs.VLLM_SM70_MTP_LEGACY_GDN_NON_SPEC_SLOT0:
        return block_table[:, 0]
    if accepted_tokens is None:
        return block_table[:, 0]
    state_offsets = torch.clamp(
        accepted_tokens.to(device=block_table.device, dtype=torch.long) - 1,
        min=0,
        max=min(num_spec, block_table.shape[1] - 1),
    )
    row_indices = torch.arange(
        block_table.shape[0], device=block_table.device, dtype=torch.long
    )
    return block_table[row_indices, state_offsets]


def build_gdn_spec_decode_state_contract(
    *,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    num_spec: int,
    spec_sequence_masks_cpu: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    current_state_block_ids: torch.Tensor | None,
    is_mamba_cache_all: bool,
    spec_state_slot_selectors: torch.Tensor | None = None,
    _преф_спек: int | None = None,
) -> GDNSpecDecodeStateContract:
    """Build the state-index/count contract consumed by active-MTP GDN.

    ``current_state_block_ids`` is authoritative for align-mode replay because
    it is materialized from the live ``mamba_state_idx`` after preprocess
    rollover. The accepted count historically also selected the committed
    speculative slot as ``num_accepted_tokens - 1`` in the recurrent kernels.
    DDTree can accept a non-linear tree path, so callers may pass
    ``spec_state_slot_selectors`` to select that slot independently.

    [FA2/SM70 10.09, ОДНОРОДНАЯ МАСКА] ``_преф_спек`` -- число ведущих истин маски, если она
    ровно префиксная (считается по процессорной копии в ``build``); тогда выборки по маске
    идут видами через ``_выбор`` без ``nonzero`` и синхронизации (FA2SM70_GDN_UNIFORM_MASK=1).
    """
    assert spec_sequence_masks_cpu.dtype == torch.bool
    assert num_accepted_tokens is not None

    def _mask_for(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == spec_sequence_masks_cpu.device:
            return spec_sequence_masks_cpu
        return spec_sequence_masks_cpu.to(tensor.device, non_blocking=True)

    block_mask = _mask_for(block_table_tensor)
    seq_mask = _mask_for(seq_lens)
    accepted_mask = _mask_for(num_accepted_tokens)
    if spec_state_slot_selectors is None:
        spec_state_slot_selectors = num_accepted_tokens
    selector_mask = _mask_for(spec_state_slot_selectors)

    if current_state_block_ids is not None:
        current_mask = _mask_for(current_state_block_ids)
        state_block_ids = current_state_block_ids[:, : num_spec + 1]
        spec_state_indices_tensor = _выбор(
            state_block_ids, current_mask, _преф_спек, True
        )
        non_spec_source = _выбор(state_block_ids, current_mask, _преф_спек, False)
        non_spec_state_indices_tensor = select_gdn_state_block_ids(
            non_spec_source,
            _выбор(num_accepted_tokens, accepted_mask, _преф_спек, False),
            num_spec,
        )
    elif is_mamba_cache_all:
        spec_state_indices_tensor = gather_gdn_state_block_ids(
            _выбор(block_table_tensor, block_mask, _преф_спек, True),
            _выбор(seq_lens, seq_mask, _преф_спек, True),
            block_size,
            num_spec + 1,
        )
        non_spec_state_indices_tensor = gather_gdn_state_block_ids(
            _выбор(block_table_tensor, block_mask, _преф_спек, False),
            _выбор(seq_lens, seq_mask, _преф_спек, False),
            block_size,
            1,
        ).squeeze(1)
    else:
        spec_state_indices_tensor = _выбор(
            block_table_tensor, block_mask, _преф_спек, True
        )[:, : num_spec + 1]
        non_spec_state_indices_tensor = select_gdn_state_block_ids(
            _выбор(block_table_tensor, block_mask, _преф_спек, False),
            _выбор(num_accepted_tokens, accepted_mask, _преф_спек, False),
            num_spec,
        )

    spec_num_accepted_tokens = _выбор(
        num_accepted_tokens, accepted_mask, _преф_спек, True
    )
    spec_state_slot_selectors = _выбор(
        spec_state_slot_selectors, selector_mask, _преф_спек, True
    )
    if os.getenv("VLLM_SM70_GDN_STATE_CONTRACT_ASSERT") == "1":
        if spec_num_accepted_tokens.numel() != spec_state_indices_tensor.shape[0]:
            raise AssertionError(
                "GDN spec state contract mismatch: accepted-token rows do "
                "not match spec state rows"
            )
        if spec_state_slot_selectors.numel() != spec_state_indices_tensor.shape[0]:
            raise AssertionError(
                "GDN spec state contract mismatch: state-selector rows do "
                "not match spec state rows"
            )
        invalid_accept = (spec_num_accepted_tokens < 1) | (
            spec_num_accepted_tokens > num_spec + 1
        )
        if torch.any(invalid_accept).item():
            raise AssertionError(
                "GDN spec state contract mismatch: num_accepted_tokens must "
                f"be in [1, {num_spec + 1}], got "
                f"{spec_num_accepted_tokens.detach().cpu().tolist()}"
            )
        invalid_selector = (spec_state_slot_selectors < 1) | (
            spec_state_slot_selectors > num_spec + 1
        )
        if torch.any(invalid_selector).item():
            raise AssertionError(
                "GDN spec state contract mismatch: spec_state_slot_selectors "
                f"must be in [1, {num_spec + 1}], got "
                f"{spec_state_slot_selectors.detach().cpu().tolist()}"
            )
        if spec_state_indices_tensor.numel() > 0:
            rows = torch.arange(
                spec_state_indices_tensor.shape[0],
                device=spec_state_indices_tensor.device,
                dtype=torch.long,
            )
            accepted_offsets = (
                spec_state_slot_selectors.to(
                    device=spec_state_indices_tensor.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
                - 1
            )
            selected_state_slots = spec_state_indices_tensor[rows, accepted_offsets]
            if torch.any(selected_state_slots == PAD_SLOT_ID).item():
                raise AssertionError(
                    "GDN spec state contract mismatch: accepted slot points "
                    "to PAD_SLOT_ID"
                )
        if current_state_block_ids is not None:
            current_mask = _mask_for(current_state_block_ids)
            active_state_ids = current_state_block_ids[current_mask, : num_spec + 1]
            if torch.any(active_state_ids == PAD_SLOT_ID).item():
                raise AssertionError(
                    "GDN spec state contract mismatch: active align-mode "
                    "state ids contain PAD_SLOT_ID"
                )

    return GDNSpecDecodeStateContract(
        spec_state_indices_tensor=spec_state_indices_tensor,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_accepted_tokens=spec_num_accepted_tokens,
        spec_state_slot_selectors=spec_state_slot_selectors,
    )


# [FA2/SM70 25.08] ОБЩИЙ СЧЁТ ГРУПП. У модели ДЕСЯТЬ отдельных KV-групп GDN (по 4-5 слоёв), и
# строитель метаданных зовётся по разу на группу: замерено 6300 вызовов на 640 шагов = ~10 за шаг,
# 9.9 мс -- больше, чем всё остальное вне прохода. При этом входные метаданные у групп РАЗЛИЧАЮТСЯ
# ТОЛЬКО таблицей блоков и слотами: маски спекуляции, счётчики, индексы токенов и начала запросов
# совпадают по построению. Считаем их ОДИН раз на шаг и раздаём; блоко-зависимые тензоры каждая
# группа по-прежнему считает СВОИ (иначе состояние читалось бы из чужих блоков -- тихая порча).
# Ключ -- тождество объектов входа (у групп это буквально один и тот же тензор).
_ОБЩЕЕ = {"ключ": None, "знач": None}
# [РЫЧАГ -- КОНСТАНТА МОДУЛЯ, А НЕ ЧТЕНИЕ НА КАЖДЫЙ ВЫЗОВ, 31.08]
# `build` исполняется на КАЖДУЮ группу KV каждого шага (у нас их десять-двенадцать), и внутри
# стояли ДВА `os.environ.get` -- то есть двадцать с лишним обращений к словарю окружения с
# разбором строки за шаг. Окружение в рантайме не меняется, поэтому читаем один раз при импорте.
import sys as _sys
_ГРАН_ДИАГ = __import__("os").environ.get("FA2SM70_GDN_GRAN_DIAG","0")=="1"
_СЧЁТ: dict = {}
# Подстановка блока-источника в spec-таблицу (лечение границы блока состояния).
# Держится на ИСТОРИИ: куда прошлый шаг записал состояние. История сверяется с ctx,
# и при любом несовпадении берётся формула без истории -- порчи быть не может.
_ГРАН_ИСТОК = __import__("os").environ.get("FA2SM70_GRAN_SRC","0")=="1"
# [ПОДСТАНОВКА КАЖДЫЙ ШАГ -- 09.09] Окно «граница рядом» считается по ПРОЦЕССОРНЫМ длинам, а
# при асинхронном планировании они отстают на шаг: окно промахивается мимо границы, и обрыв
# возвращается. Комментарий ниже (строка про FA2SM70_GRAN_ALWAYS) обещал этот рычаг, но в коде
# его НЕ БЫЛО -- только на словах. Здесь он заведён: окно снимается, подстановка идёт всегда.
# Цена -- работа на каждом шаге вместо ~16 из 4096; мерить парно.
_ГРАН_ВСЕГДА = __import__("os").environ.get("FA2SM70_GRAN_ALWAYS","0")=="1"
# Подстановка блока-ПРИЁМНИКА: состояние на КОНЕЦ блока обязано лечь в сам блок, иначе
# префикс-кэш переиспользует недосчитанное состояние (спекулятивное ядро кладёт в блок
# состояние ПЕРВОЙ позиции шага, а не последней позиции блока).
_ГРАН_ПРИЁМ = __import__("os").environ.get("FA2SM70_GRAN_DST","0")=="1"
_ГРАН_КОНВ0 = __import__("os").environ.get("FA2SM70_GRAN_CONV0","0")=="1"
# Пара указателей для СВЁРТКИ в спекулятивной ветке: читать из блока опоры прошлого шага,
# писать в блок опоры текущего. Ровно то, что обычный декод делает через initial_state_idx
# и block_idx_last_scheduled_token, а спекулятивный не делал вовсе.
_ГРАН_КОНВ = __import__("os").environ.get("FA2SM70_GRAN_CONV","0")=="1"
# ПАРА УКАЗАТЕЛЕЙ СВЁРТКЕ НА КАЖДОМ ШАГЕ (FA2SM70_GRAN_CONV_ALL=1). Замер прибором
# запаса конца: сама УСЛОВНОСТЬ пары и есть возмущение -- у границы свёртка шла
# парой указателей, вне границы одним слотом, и переключение стоило разброса 8.64
# нат против пола 0.35. Пара на КАЖДОМ шаге: цена перехода +0.03 нат, разброс 0.61
# против пола 0.60 -- граница становится прозрачной. Здесь строится ТОЛЬКО пара,
# без подстановки SSM-столбца и клона истории (их безусловность стоила ~10 % декода).
_ГРАН_КОНВ_ВСЕГДА = __import__("os").environ.get("FA2SM70_GRAN_CONV_ALL","0")=="1"
# Прибор и фальсификатор к разбору падения 06.09 (Xid 13 на боевом): считать выходы за
# границу и уметь СНЯТЬ зажим, чтобы доказать, что падение шло именно оттуда.
_ГРАН_КОНВ_ДИАГ = __import__("os").environ.get("FA2SM70_GRAN_CONV_DIAG","0")=="1"
_ГРАН_КОНВ_БЕЗ_ЗАЖИМА = __import__("os").environ.get("FA2SM70_GRAN_CONV_NOCLAMP","0")=="1"
# Бисекция пары: 1 = читать оттуда же, куда пишем (без истории прошлого шага).
_ГРАН_КОНВ_БЕЗ_ИСТОРИИ = __import__("os").environ.get("FA2SM70_GRAN_CONV_NOHIST","0")=="1"
# Бисекция: отдать ядру ЧУЖУЮ таблицу блоков напрямую, без нашего постоянного буфера.
# Если падение уходит -- виноват буфер (устаревшие строки/столбцы), если остаётся -- сам путь.
_ГРАН_КОНВ_СЫРАЯ = __import__("os").environ.get("FA2SM70_GRAN_CONV_RAW","0")=="1"
_СПЕК_ПО_CTX = __import__("os").environ.get("FA2SM70_GDN_SPEC_CTX","0")=="1"
# [ГОНКА, 07.09] Все копии метаданных сюда идут асинхронно. Если источник --
# ЗАКРЕПЛЁННЫЙ буфер хоста, переиспользуемый на следующем шаге, питон вправе
# переписать его до того, как DMA прошлого шага завершилась: порча метаданных ->
# негодные номера блоков -> обращение за буфер в чужом ядре. Улика в пользу гонки:
# при CUDA_LAUNCH_BLOCKING=1 стенд прошёл 1400 запросов без падения.
# Рычаг делает копии синхронными, чтобы проверить это ЗАМЕРОМ, а не рассуждением.
_НЕБЛОК = _os.environ.get("FA2SM70_GDN_ASYNC_COPY", "1") == "1"
# [ОДНОРОДНАЯ МАСКА -- 10.09, записка 25 §282]
# Выборка GPU-тензора булевой маской (`t[~spec_sequence_masks]`) внутри зовёт `nonzero`, а это
# СИНХРОНИЗАЦИЯ с картой. Трасса со стеками: в `build` 7 `nonzero` и 7 `item` за шаг, и вместе
# с `index` они стоят 1.3 мс из 3.85 -- то есть треть строителя и четверть всей обвязки.
# А в устойчивом декоде маска ОДНОРОДНА: все запросы спекулятивные, значит `~маска` не выбирает
# НИЧЕГО, а `маска` выбирает ВСЁ. Тогда выборка заменяется видом -- бесплатно и без синхронизации.
# Однородность проверяется по ПРОЦЕССОРНОЙ копии маски (она тут же и считается), поэтому сама
# проверка синхронизации не требует. Значения тождественны по построению.
# Умолчание ВЫКЛЮЧЕНО: venv общий с боевым, и новое поведение не должно приезжать
# туда само при первом же перезапуске. Замер: строитель 1.63 -> 1.39 мс (-16 %),
# пять синхронизаций за шаг убраны, приёмка и гейт 391 не тронуты.
_ОДНОРОДН = _os.environ.get("FA2SM70_GDN_UNIFORM_MASK", "0") == "1"


def _выбор(т, маска, преф, брать_спек: bool):
    """Выборка по булевой маске без `nonzero`, когда маска -- ПРЕФИКС из истин.

    `преф` -- число ведущих истин, если маска ровно префиксная, иначе None. Тогда выборка по
    маске это `т[:преф]`, а по инверсии `т[преф:]` -- виды, без выделения и без синхронизации.
    ПОЧЕМУ ПРЕФИКС, А НЕ «ВСЯ ИСТИННА»: маска строится по ДОПОЛНЕННОМУ батчу, у строк-
    заполнителей `num_decode_draft_tokens = -1`, то есть ложь. «Вся истинна» не бывает никогда --
    первая редакция этой правки из-за того и не срабатывала (замерено: строитель не подешевел).
    Настоящие запросы идут первыми, поэтому префиксность -- обычный случай.
    """
    if _ОДНОРОДН and преф is not None:
        return т[:преф] if брать_спек else т[преф:]
    return т[маска] if брать_спек else т[~маска]

_ДЕЛИТЬ_ОБЩЕЕ = _os.environ.get("FA2SM70_GDN_SHARE", "1") == "1"


def _общий_ключ(m, num_accepted_tokens):
    """Ключ шага. ВНИМАНИЕ: id() безопасен ТОЛЬКО пока объект жив.

    Python переиспользует адреса после сборки мусора, поэтому новый объект может получить id
    старого -- и кэш отдаст ЧУЖИЕ тензоры. Отказа не будет, будет тихая порча. Лечение: вместе с
    ключом держим ЖИВЫЕ ССЫЛКИ на те самые объекты (`_ОБЩЕЕ["якорь"]`), тогда их адреса не могут
    быть переиспользованы, пока запись в кэше действительна.
    """
    # РАЗЛИЧИТЕЛЬ ШАГА ОБЯЗАТЕЛЕН. Раннер переиспользует ОДНИ И ТЕ ЖЕ буферы каждый шаг, поэтому
    # id() совпадает между шагами, а `num_actual_tokens`/`num_reqs` в установившемся декоде
    # постоянны -- кэш отдал бы УСТАРЕВШИЕ индексы блоков (тихая порча, не отказ). `max_seq_len`
    # растёт на каждом шаге декода и одинаков у всех групп внутри шага -- это и есть нужный ключ.
    return (id(m.query_start_loc), id(m.seq_lens), m.num_actual_tokens,
            m.num_reqs, int(m.max_seq_len), id(num_accepted_tokens))


# [FA2/SM70 25.08] ОБЩИЕ БУФЕРЫ ГРУПП. У модели ДЕСЯТЬ KV-групп GDN, и все поля метаданных,
# кроме ИНДЕКСОВ СОСТОЯНИЙ, у них совпадают побитово (они зависят от ДЛИН, а не от таблицы
# блоков). Прежде каждая группа держала свои буферы и `update_block_table` копировал в них по
# двенадцать полей -- замер фазомером: 2.43 мс на шаг при девяти обновлениях.
#
# Общий буфер на все группы законен ИМЕННО потому, что общий и при ЗАХВАТЕ графа: каждая группа
# запекает один и тот же адрес, и заполняет его тот, кто строит первым. Прежняя авария (гейт
# «39» вместо «391») была ОБРАТНОЙ: группа отдавала ЧУЖОЙ буфер, которого её граф не видел.
# Здесь адрес один у всех дорог -- и при захвате, и при повторе.
_ОБЩИЕ_БУФЕРЫ: dict = {}


def _общий_буфер(имя, форма, тип, device):
    ключ = (имя, tuple(форма), тип, str(device))
    т = _ОБЩИЕ_БУФЕРЫ.get(ключ)
    if т is None:
        т = _ОБЩИЕ_БУФЕРЫ[ключ] = torch.zeros(форма, dtype=тип, device=device)
    return т


_ОБЩИЕ_ВКЛ = _os.environ.get("FA2SM70_GDN_SHARED_BUF", "1") == "1"
# [ЖИЗНЬ БУФЕРА, 08.09] Растущие буферы отпускать нельзя: их адрес запечён в графе.
_МАСКА_БУФ = int(_os.environ.get("FA2SM70_KEEP_BUFS", "0") or 0)
_ДЕРЖАТЬ_GDN = bool(_МАСКА_БУФ & 9)          # 1=всё, 8=оба буфера gdn
# Разделение на два, чтобы назвать ВИНОВНИКА, а не держать оба:
#   16 = только таблица блоков (_бs), 32 = только буфер индексов состояния (_буф_ssm)
_ДЕРЖАТЬ_БЛОКИ = bool(_МАСКА_БУФ & (9 | 16))
_ДЕРЖАТЬ_SSM = bool(_МАСКА_БУФ & (9 | 32))
_ПЕНСИЯ_GDN: list = []


def _отставить_gdn(*т):
    if True:
        for x in т:
            if x is not None:
                _ПЕНСИЯ_GDN.append(x)


class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        assert isinstance(kv_cache_spec, MambaSpec)
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.speculative_config = vllm_config.speculative_config
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _resolve_gdn_prefill_backend,
        )

        self.gdn_prefill_backend: Literal[
            "triton", "flashinfer", "cutedsl", "flashqla_sm70"
        ]
        _, self.gdn_prefill_backend = _resolve_gdn_prefill_backend(vllm_config)

        if self.speculative_config:
            assert self.speculative_config.num_speculative_tokens is not None
            self.num_spec: int = self.speculative_config.num_speculative_tokens
            self.num_spec_state_tokens: int = (
                self.speculative_config.num_speculative_state_tokens()
            )
        else:
            self.num_spec = 0
            self.num_spec_state_tokens = 0
        self.use_spec_decode: bool = self.num_spec > 0
        self._init_reorder_batch_threshold(1, self.use_spec_decode)

        self.use_full_cuda_graph: bool = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self._потолок_запросов = int(getattr(getattr(vllm_config, 'scheduler_config', None),
                                             'max_num_seqs', 0) or 0)
        self.decode_cudagraph_max_bs: int = (
            self.vllm_config.scheduler_config.max_num_seqs
            * (self.num_spec_state_tokens + 1)
        )
        if self.compilation_config.max_cudagraph_capture_size is not None:
            self.decode_cudagraph_max_bs = min(
                self.decode_cudagraph_max_bs,
                self.compilation_config.max_cudagraph_capture_size,
            )

        # [СВЁРТКА У ГРАНИЦЫ, 04.09] ПОСТОЯННЫЕ буферы под таблицу и пару указателей.
        # Новые тензоры каждый шаг полный граф не видит: он читает адреса, снятые при
        # захвате, и обращение уходит в освобождённую память -- падало как illegal memory
        # access в чужом ядре. Ширину берём с запасом под таблицу блоков любой группы.
        _шир_бл = cdiv(vllm_config.model_config.max_model_len,
                       kv_cache_spec.block_size) + self.num_spec + 1
        self._conv_блоки_буф = torch.empty(
            (self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1),
             _шир_бл), dtype=torch.int32, device=device)
        # [БУФЕР ЧТЕНИЯ -- ВСЕГДА НУЛИ, 08.09] Колонка чтения спекулятивной свёртки равна
        # нулю ПО ПОСТРОЕНИЮ (таблица выровнена по контексту, состояние лежит в колонке 0):
        # в `build` она считалась как `torch.zeros_like(...)` и копировалась в буфер ДВАЖДЫ
        # за шаг. Заводим буфер нулями один раз -- и не трогаем вовсе: минус шесть запусков
        # ядер на каждом шаге в горячем пути (замер положительным контролем: хозяйская
        # работа стоит 1:1, §218).
        self._conv_чт_буф = torch.zeros(
            self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1),
            dtype=torch.int32, device=device)
        self._conv_зап_буф = torch.empty_like(self._conv_чт_буф)
        # [ДЕРЕВО, 05.09-3] Буферы off и nacc_ssm рождаются ЗДЕСЬ -- ДО захвата графов.
        # Ленивые буферы появлялись на первом декодном шаге, ПОЗЖЕ захвата: поле
        # num_accepted_ssm при захвате было None, ядро запекалось с адресом обычного nacc,
        # и сдвиг чтения не действовал никогда (дозор: nacc=2, off=2, а слой видел 2).
        if self.use_spec_decode:
            self._nacc_ssm_буф = torch.ones(
                self.decode_cudagraph_max_bs, dtype=torch.int32, device=device)
            # Свой буфер и для СВЁРТКИ: протокол-реплей показал, что захваченный графом
            # nacc-адрес живёт то с лагом на шаг, то нулём после первого принятия ветви --
            # клубок адресов обходится ЯВНЫМ буфером, созданным до захвата.
            self._nacc_conv_буф = torch.ones(
                self.decode_cudagraph_max_bs, dtype=torch.int32, device=device)
            try:
                from vllm.model_executor.models.qwen3_next import ДЕРЕВО_OFF as _ДО0
                if _ДО0.get("буф") is None:
                    _ДО0["буф"] = torch.zeros(
                        vllm_config.scheduler_config.max_num_seqs,
                        dtype=torch.int32, device=device)
            except Exception:
                pass
        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec_state_tokens + 1),
            dtype=torch.int32,
            device=device,
        )
        # РЕЖИМ 'all': на запрос приходится не один слот, а СТОЛБЕЦ блоков, поэтому
        # постоянный буфер под CUDA-граф обязан быть двумерным. Если оставить его
        # одномерным и подсовывать графу свежий тензор, граф запомнит ЧУЖОЙ адрес --
        # отказа не будет, будет тихая порча (тот же класс, что мы уже ловили гейтом).
        self.mamba_all = vllm_config.cache_config.mamba_cache_mode == "all"
        # ШИРИНА БУФЕРА СЧИТАЕТСЯ ТОЙ ЖЕ ФОРМУЛОЙ, ЧТО И ТАБЛИЦА БЛОКОВ В ДВИЖКЕ.
        # ОТКАЗ, КОТОРЫЙ ЭТО ЛЕЧИТ (боевой :8085, 28.08 23:40, аптайм 4.5 ч):
        #   RuntimeError: The size of tensor a (64) must match the size of tensor b (67)
        #   at non-singleton dimension 1  <- gdn_attn.py:723, copy_ в постоянный буфер.
        # Движок (gpu_model_runner._init_..., «mamba_blocks_per_req») даёт таблице ширину
        #   cdiv(max_model_len, block_size) + num_speculative_blocks
        # при включённом префикс-кэше, а здесь стояло только cdiv(...). Разница -- РОВНО
        # число блоков спекуляции (3), поэтому отказ ждал первого шага, где спекулятивных
        # декодов НЕТ (первый шаг после префилла): только там источник копируется целиком.
        # Класс тот же, что в отчёте 27.08 §4.1: «величина, ВЫВЕДЕННАЯ по формуле, вместо
        # взятой у того, кто её задал». Формулу дублируем ДОСЛОВНО, чтобы расхождение не
        # воскресло: одна и та же величина в двух местах обязана считаться одинаково.
        _блоков = cdiv(vllm_config.model_config.max_model_len, kv_cache_spec.block_size)
        _спек = getattr(kv_cache_spec, "num_speculative_blocks", 0)
        self.mamba_max_blocks = (
            max(_блоков,
                (_блоков if vllm_config.cache_config.enable_prefix_caching else 1) + _спек)
            if self.mamba_all
            else 1
        )
        # Общий буфер при FA2SM70_GDN_SHARED_BUF=1, иначе свой на группу (прежнее поведение).
        _вб = (_общий_буфер if _ОБЩИЕ_ВКЛ
               else (lambda имя, ф, т, d: torch.empty(ф, dtype=т, device=d)))
        self.non_spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.mamba_max_blocks)
            if self.mamba_all
            else (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self.block_idx_last_computed_token = _вб(
            "blk_last_comp", (self.decode_cudagraph_max_bs,), torch.int32, device)
        self.block_idx_last_scheduled_token = _вб(
            "blk_last_sched", (self.decode_cudagraph_max_bs,), torch.int32, device)
        self.spec_sequence_masks: torch.Tensor = _вб(
            "spec_masks", (self.decode_cudagraph_max_bs,), torch.bool, device)
        self.spec_token_indx: torch.Tensor = _вб(
            "spec_tok",
            (self.decode_cudagraph_max_bs * (self.num_spec_state_tokens + 1),),
            torch.int32, device)
        self.non_spec_token_indx: torch.Tensor = _вб(
            "nonspec_tok",
            (self.decode_cudagraph_max_bs * (self.num_spec_state_tokens + 1),),
            torch.int32, device)
        self._spec_token_indx_initialized_size = 0
        self.spec_query_start_loc: torch.Tensor = _вб(
            "spec_qsl", (self.decode_cudagraph_max_bs + 1,), torch.int32, device)
        self.non_spec_query_start_loc: torch.Tensor = _вб(
            "nonspec_qsl", (self.decode_cudagraph_max_bs + 1,), torch.int32, device)
        self.num_accepted_tokens: torch.Tensor = _вб(
            "nacc", (self.decode_cudagraph_max_bs,), torch.int32, device)
        self.spec_state_slot_selectors: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs,),
            dtype=torch.int32,
            device=device,
        )
        self._ddtree_fast_common_buffers: _GDNDdTreeFastCommonBuffers | None = None
        self._ddtree_fast_tail_key: tuple[int, int, int] | None = None
        if (
            self.use_spec_decode
            and envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP
            and _dflash_ddtree_gdn_shared_common_enabled()
        ):
            self._ddtree_fast_common_buffers = _get_ddtree_gdn_fast_common_buffers(
                device,
                self.decode_cudagraph_max_bs,
                self.num_spec_state_tokens + 1,
            )
        if self.use_spec_decode and envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP:
            placeholder_rows = max(
                1, min(self.num_spec_state_tokens + 1, self.decode_cudagraph_max_bs)
            )
            common_buffers = self._ddtree_fast_common_buffers
            self.non_spec_query_start_loc[: placeholder_rows + 1].fill_(0)
            self.non_spec_state_indices_tensor[:placeholder_rows].fill_(PAD_SLOT_ID)
            spec_query_start_loc = (
                common_buffers.spec_query_start_loc
                if common_buffers is not None
                else self.spec_query_start_loc
            )
            spec_sequence_masks = (
                common_buffers.spec_sequence_masks
                if common_buffers is not None
                else self.spec_sequence_masks
            )
            spec_token_indx = (
                common_buffers.spec_token_indx
                if common_buffers is not None
                else self.spec_token_indx
            )
            non_spec_token_indx = (
                common_buffers.non_spec_token_indx
                if common_buffers is not None
                else self.non_spec_token_indx
            )
            num_accepted_tokens = (
                common_buffers.num_accepted_tokens
                if common_buffers is not None
                else self.num_accepted_tokens
            )
            spec_state_slot_selectors = (
                common_buffers.spec_state_slot_selectors
                if common_buffers is not None
                else self.spec_state_slot_selectors
            )
            spec_query_start_loc[: placeholder_rows + 1].fill_(0)
            self.spec_state_indices_tensor[:placeholder_rows].fill_(PAD_SLOT_ID)
            spec_sequence_masks[:placeholder_rows].fill_(False)
            spec_token_indx[:placeholder_rows].copy_(
                torch.arange(
                    placeholder_rows,
                    dtype=torch.int32,
                    device=device,
                )
            )
            if common_buffers is not None:
                common_buffers.token_index_initialized_size = max(
                    common_buffers.token_index_initialized_size,
                    placeholder_rows,
                )
            else:
                self._spec_token_indx_initialized_size = placeholder_rows
            non_spec_token_indx[:0].fill_(0)
            num_accepted_tokens[:placeholder_rows].fill_(1)
            spec_state_slot_selectors[:placeholder_rows].fill_(1)
            register_gdn_spec_metadata_tensors(
                self.layer_names,
                (
                    self.non_spec_query_start_loc[: placeholder_rows + 1],
                    self.non_spec_state_indices_tensor[:placeholder_rows],
                    spec_query_start_loc[: placeholder_rows + 1],
                    self.spec_state_indices_tensor[:placeholder_rows],
                    spec_token_indx[:placeholder_rows],
                    non_spec_token_indx[:0],
                    spec_sequence_masks[:placeholder_rows],
                    num_accepted_tokens[:placeholder_rows],
                    spec_state_slot_selectors[:placeholder_rows],
                ),
            )

    # [FA2/SM70 25.08] ВЫКЛЮЧЕНО ПО ПРОВАЛУ ГЕЙТА. Механизм написан (см. update_block_table ниже) и
    # даёт правильную форму, но гейт «17*23» вернул «39» вместо «391» -- ТИХАЯ ПОРЧА, а не отказ.
    # Значит пересборка индексов состояний под чужую таблицу расходится с `build` в чём-то, что
    # формой не ловится (подозрение: паддинг постоянных буферов и ветка non_spec). Включать только
    # после ПОБИТОВОЙ сверки метаданных обеих дорог на живом шаге, а не «по виду».
    supports_update_block_table: bool = (
        _os.environ.get("FA2SM70_GDN_UPDATE", "0") == "1"
    )  # включается только вместе со сверкой FA2SM70_META_CHECK

    def _индексы_состояний(
        self,
        block_table_tensor,
        aligned_block_table,
        spec_sequence_masks,
        mamba_all: bool,
        только_спекуляция: bool,
    ):
        """ЕДИНОЕ тело для build и update_block_table. Два тела здесь уже стоили подъёма.

        [ПОРТ 07.2026] Ширина спекулятивной таблицы -- num_spec_state_tokens + 1 (так её
        заводит новый upstream); без DDTree это то же num_spec + 1.
        """
        if spec_sequence_masks is None:
            spec_state = None
            non_spec_state = (
                block_table_tensor if mamba_all else block_table_tensor[:, 0]
            )
            return spec_state, non_spec_state
        _src = aligned_block_table if mamba_all else block_table_tensor
        if только_спекуляция:
            return _src[:, : self.num_spec_state_tokens + 1], None
        return (
            _src[spec_sequence_masks, : self.num_spec_state_tokens + 1],
            block_table_tensor[~spec_sequence_masks]
            if mamba_all
            else block_table_tensor[~spec_sequence_masks, 0],
        )

    def update_block_table(self, metadata, blk_table, slot_mapping):
        """Пересборка ТОЛЬКО блоко-зависимых полей под таблицу другой группы.

        ЗАЧЕМ. У модели ДЕСЯТЬ KV-групп GDN, и без этого движок звал `build` по разу на группу --
        замерено 10 вызовов за шаг, 9.9 мс (наш строитель внимания -- 0.07). Всё, кроме индексов
        состояний, у групп совпадает: маски, счётчики, индексы токенов, начала запросов и индексы
        блоков (они зависят от ДЛИН, а не от таблицы). Механизм штатный -- ровно так делает
        Mamba2 (`mamba_attn.py`), у GDN он просто не был реализован.
        """
        import copy as _copy

        новое = _copy.copy(metadata)
        режим = self.vllm_config.cache_config.mamba_cache_mode
        mamba_all = режим == "all"
        seq_lens = metadata.seq_lens_для_обновления
        block_table_tensor = mamba_get_block_table_tensor(
            blk_table, seq_lens, self.kv_cache_spec, режим
        )
        aligned = (
            mamba_get_block_table_tensor(blk_table, seq_lens, self.kv_cache_spec, "align")
            if mamba_all
            else None
        )
        только_спек = metadata.num_prefills == 0 and metadata.num_decodes == 0
        # [СВЁРТКА У ГРАНИЦЫ] ТАБЛИЦА -- СВОЯ У ГРУППЫ, УКАЗАТЕЛИ -- ОБЩИЕ.
        # Колонки чтения и записи зависят только от ДЛИН, поэтому считаются один раз в
        # `build` и переносятся копией метаданных. А сама таблица блоков у каждой группы
        # своя, и без пересборки девять групп из десяти читали бы состояние по таблице
        # чужой группы -- ровно та тихая порча, о которой предупреждает закон выше
        # (совпадение значений не значит правильности). Замер без пересборки: полных
        # ответов 0 из 8, ответы вырождались до 24 токенов.
        if ((_ГРАН_КОНВ or _ГРАН_КОНВ_ВСЕГДА) and mamba_all
                and getattr(metadata, "spec_conv_чт", None) is not None):
            _пг = block_table_tensor
            if metadata.spec_sequence_masks is not None and not только_спек:
                _мг = metadata.spec_sequence_masks[: _пг.shape[0]]
                _пг = _пг[_мг]
            _нг2 = min(int(metadata.spec_conv_чт.shape[0]), _пг.shape[0],
                       self._conv_блоки_буф.shape[0])
            _вш2 = min(_пг.shape[1], self._conv_блоки_буф.shape[1])
            if _нг2 > 0 and _вш2 > 0:
                self._conv_блоки_буф[:_нг2, :_вш2].copy_(_пг[:_нг2, :_вш2])
                if metadata.spec_state_indices_tensor is not None:
                    _пад2 = (metadata.spec_state_indices_tensor[:_нг2, 0] == PAD_SLOT_ID)
                    self._conv_блоки_буф[:_нг2][_пад2] = PAD_SLOT_ID
                новое.spec_conv_блоки = self._conv_блоки_буф[:_нг2, :_вш2]
            else:
                новое.spec_conv_блоки = None
        spec_state, non_spec_state = self._индексы_состояний(
            block_table_tensor, aligned, metadata.spec_sequence_masks,
            mamba_all, только_спек,
        )
        # ПОСТОЯННЫЕ БУФЕРЫ -- СВОИ У ЭТОЙ ГРУППЫ (её граф захватил именно эти адреса).
        # ЗАКОН, ДОБЫТЫЙ ОШИБКОЙ (25.08): СОВПАДЕНИЕ ЗНАЧЕНИЙ НЕ ЗНАЧИТ ПРАВИЛЬНОСТИ ПРИ ГРАФАХ.
        # Первая редакция копировала только индексы состояний, а остальные поля отдавала ЧУЖИЕ
        # (буферы соседней группы). Побитовая сверка 378 шагов показала НОЛЬ расхождений по
        # значениям -- и всё равно гейт вернул «39» вместо «391»: граф читает АДРЕСА своих
        # буферов, а их никто не заполнил. Поэтому копируем ВСЕ поля, которые заполняет `build`.
        # Условия и добивка PAD_SLOT_ID повторяют ветку `build` ОДИН В ОДИН: разойдись они --
        # получим тихую порчу на паддинге, а не отказ.
        if (
            self.use_full_cuda_graph
            and metadata.num_prefills == 0
            and metadata.num_decodes == 0
            and metadata.num_spec_decodes > 0
            and metadata.num_spec_decodes <= self.decode_cudagraph_max_bs
            and metadata.num_spec_decode_tokens <= self.decode_cudagraph_max_bs
            and spec_state is not None
            and metadata.spec_state_indices_tensor is not None
        ):
            n = metadata.num_spec_decodes
            batch_size = metadata.spec_state_indices_tensor.shape[0]
            self.spec_state_indices_tensor[:n].copy_(spec_state[:n], non_blocking=_НЕБЛОК)
            spec_state = self.spec_state_indices_tensor[:batch_size]
            spec_state[n:].fill_(PAD_SLOT_ID)

            # ОСТАЛЬНЫЕ ПОЛЯ -- В СВОИ БУФЕРЫ, порядок и добивка как в `build`.
            # При ОБЩИХ буферах копировать НЕЧЕГО: `новое` -- поверхностная копия метаданных
            # группы-строителя, и её поля УЖЕ указывают на тот самый общий буфер, который
            # запёк граф этой группы. Копия была бы тензора в себя.
            _м = metadata
            if _м.spec_sequence_masks is not None and not _ОБЩИЕ_ВКЛ:
                self.spec_sequence_masks[:n].copy_(_м.spec_sequence_masks[:n], non_blocking=_НЕБЛОК)
                новое.spec_sequence_masks = self.spec_sequence_masks[:batch_size]
                новое.spec_sequence_masks[n:].fill_(False)
            if _м.non_spec_token_indx is not None and not _ОБЩИЕ_ВКЛ:
                _к = _м.non_spec_token_indx.size(0)
                self.non_spec_token_indx[:_к].copy_(_м.non_spec_token_indx, non_blocking=_НЕБЛОК)
                новое.non_spec_token_indx = self.non_spec_token_indx[:_к]
            if _м.spec_token_indx is not None and not _ОБЩИЕ_ВКЛ:
                _к = _м.spec_token_indx.size(0)
                self.spec_token_indx[:_к].copy_(_м.spec_token_indx, non_blocking=_НЕБЛОК)
                новое.spec_token_indx = self.spec_token_indx[:_к]
            if _м.spec_query_start_loc is not None and not _ОБЩИЕ_ВКЛ:
                self.spec_query_start_loc[: n + 1].copy_(
                    _м.spec_query_start_loc[: n + 1], non_blocking=_НЕБЛОК
                )
                _хвост = _м.spec_query_start_loc[n]
                новое.spec_query_start_loc = self.spec_query_start_loc[: batch_size + 1]
                новое.spec_query_start_loc[n + 1 :].fill_(_хвост)
            if _м.num_accepted_tokens is not None and not _ОБЩИЕ_ВКЛ:
                self.num_accepted_tokens[:n].copy_(_м.num_accepted_tokens[:n], non_blocking=_НЕБЛОК)
                новое.num_accepted_tokens = self.num_accepted_tokens[:batch_size]
                новое.num_accepted_tokens[n:].fill_(1)
                if getattr(_м, "num_accepted_ssm", None) is not None:
                    # ОДИН БУФЕР С build. Второй буфер здесь прибивал граф к адресу,
                    # который build не обновляет: SSM-nacc замерзал на значении захвата,
                    # и чтение состояния шло из колонки якоря при ЛЮБОМ m -- источник
                    # сплошных «!» при включённом OFF.
                    _бs = getattr(self, "_nacc_ssm_буф", None)
                    if _бs is None or _бs.shape[0] < batch_size:
                        _отставить_gdn(_бs) if _ДЕРЖАТЬ_БЛОКИ else None
                        _бs = self._nacc_ssm_буф = torch.ones(
                            max(batch_size, self.decode_cudagraph_max_bs),
                            dtype=_м.num_accepted_ssm.dtype,
                            device=_м.num_accepted_ssm.device)
                    if _м.num_accepted_ssm.data_ptr() != _бs.data_ptr():
                        _бs[:n].copy_(_м.num_accepted_ssm[:n], non_blocking=_НЕБЛОК)
                    _бs[n:batch_size].fill_(1)
                    новое.num_accepted_ssm = _бs[:batch_size]
                if getattr(_м, "num_accepted_conv", None) is not None:
                    _бк2 = getattr(self, "_nacc_conv_буф", None)
                    if _бк2 is not None and _бк2.shape[0] >= batch_size:
                        if _м.num_accepted_conv.data_ptr() != _бк2.data_ptr():
                            _бк2[:n].copy_(_м.num_accepted_conv[:n], non_blocking=_НЕБЛОК)
                        _бк2[n:batch_size].fill_(1)
                        новое.num_accepted_conv = _бк2[:batch_size]
        # ВТОРАЯ ВЕТКА БУФЕРОВ -- ЧИСТЫЙ ДЕКОД (без спекуляции). Её пропуск дал ПУСТЫЕ ОТВЕТЫ в
        # конфигурации боевого: значения совпадали, но граф читал незаполненные буферы группы.
        # Зеркалим `build` один в один, включая добивку.
        if (
            self.use_full_cuda_graph
            and metadata.num_prefills == 0
            and metadata.num_spec_decodes == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
            and non_spec_state is not None
        ):
            _д = metadata.num_decodes
            _bs = (metadata.non_spec_state_indices_tensor.shape[0]
                   if metadata.non_spec_state_indices_tensor is not None else _д)
            self.non_spec_state_indices_tensor[:_д].copy_(non_spec_state[:_д], non_blocking=_НЕБЛОК)
            non_spec_state = self.non_spec_state_indices_tensor[:_bs]
            non_spec_state[_д:].fill_(PAD_SLOT_ID)

            if (mamba_all and not _ОБЩИЕ_ВКЛ
                    and metadata.block_idx_last_computed_token is not None):
                self.block_idx_last_computed_token[:_д].copy_(
                    metadata.block_idx_last_computed_token[:_д], non_blocking=_НЕБЛОК)
                новое.block_idx_last_computed_token = self.block_idx_last_computed_token[:_bs]
                новое.block_idx_last_computed_token[_д:].fill_(0)
                self.block_idx_last_scheduled_token[:_д].copy_(
                    metadata.block_idx_last_scheduled_token[:_д], non_blocking=_НЕБЛОК)
                новое.block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[:_bs]
                новое.block_idx_last_scheduled_token[_д:].fill_(0)

            if metadata.non_spec_query_start_loc is not None and not _ОБЩИЕ_ВКЛ:
                self.non_spec_query_start_loc[: _д + 1].copy_(
                    metadata.non_spec_query_start_loc[: _д + 1], non_blocking=_НЕБЛОК)
                _хв = metadata.non_spec_query_start_loc[_д]
                новое.non_spec_query_start_loc = self.non_spec_query_start_loc[: _bs + 1]
                новое.non_spec_query_start_loc[_д + 1 :].fill_(_хв)

        новое.spec_state_indices_tensor = spec_state
        новое.non_spec_state_indices_tensor = non_spec_state
        return новое

    def _build_fast_pure_ddtree_full_graph(
        self,
        *,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None,
        spec_state_slot_selectors: torch.Tensor | None,
        num_decode_draft_tokens_cpu: torch.Tensor | None,
        current_state_block_ids: torch.Tensor | None,
        ddtree_parent_ids: torch.Tensor | None,
        ddtree_num_tree_tokens_cpu: torch.Tensor | None,
        for_cudagraph_capture: bool,
        fast_build_epoch: int | None,
        metadata_profile: bool,
        metadata_profile_t0: float,
    ) -> GDNAttentionMetadata | None:
        def _miss(reason: str) -> GDNAttentionMetadata | None:
            if os.getenv("VLLM_DFLASH_DDTREE_FAST_BUILD_DEBUG", "0") != "1":
                return None
            count = getattr(self, "_ddtree_fast_build_miss_count", 0)
            if count >= 8:
                return None
            self._ddtree_fast_build_miss_count = count + 1
            logger.info(
                "DFLASH_DDTREE_METADATA_PROFILE gdn_build_fast_miss "
                "reason=%s full_graph=%s spec=%s core_op=%s "
                "has_parent=%s has_tree_tokens=%s has_accept=%s "
                "has_draft=%s has_state_ids=%s",
                reason,
                self.use_full_cuda_graph,
                self.use_spec_decode,
                envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP,
                ddtree_parent_ids is not None,
                ddtree_num_tree_tokens_cpu is not None,
                num_accepted_tokens is not None,
                num_decode_draft_tokens_cpu is not None,
                current_state_block_ids is not None,
            )
            return None

        if (
            for_cudagraph_capture
            or not self.use_full_cuda_graph
            or not self.use_spec_decode
            or not envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP
            or ddtree_parent_ids is None
            or ddtree_num_tree_tokens_cpu is None
            or num_accepted_tokens is None
            or num_decode_draft_tokens_cpu is None
        ):
            return _miss("missing_required")

        m = common_attn_metadata
        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        num_reqs = query_start_loc_cpu.numel() - 1
        if num_reqs != 1:
            return None
        if (
            ddtree_parent_ids.ndim != 2
            or ddtree_parent_ids.shape[0] < 1
            or ddtree_num_tree_tokens_cpu.ndim != 1
            or ddtree_num_tree_tokens_cpu.numel() < 1
            or num_accepted_tokens.ndim != 1
            or num_accepted_tokens.numel() < 1
            or num_decode_draft_tokens_cpu.ndim != 1
            or num_decode_draft_tokens_cpu.numel() < 1
            or (
                current_state_block_ids is not None
                and (
                    current_state_block_ids.ndim != 2
                    or current_state_block_ids.shape[0] < 1
                )
            )
        ):
            return _miss("shape")

        query_len = int((query_start_loc_cpu[1] - query_start_loc_cpu[0]).item())
        tree_tokens = int(ddtree_num_tree_tokens_cpu[0].item())
        if (
            int(query_start_loc_cpu[0].item()) != 0
            or query_len <= 1
            or tree_tokens + 1 != query_len
            or int(num_decode_draft_tokens_cpu[0].item()) <= 0
        ):
            return _miss("not_pure_ddtree")

        width = self.num_spec_state_tokens + 1
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
        is_mamba_cache_all = self.vllm_config.cache_config.mamba_cache_mode == "all"
        if current_state_block_ids is not None:
            fast_path_state_block_ids = current_state_block_ids[:1, :width]
        elif is_mamba_cache_all:
            seq_lens_cpu_for_state = m._seq_lens_cpu
            if seq_lens_cpu_for_state is not None:
                seq_len = int(seq_lens_cpu_for_state[0].item())
                start_idx = max((seq_len - 1) // self.kv_cache_spec.block_size, 0)
                end_idx = start_idx + width
                if end_idx > block_table_tensor.shape[1]:
                    return _miss("state_block_table_capacity")
                fast_path_state_block_ids = block_table_tensor[:1, start_idx:end_idx]
            else:
                fast_path_state_block_ids = gather_gdn_state_block_ids(
                    block_table_tensor[:1],
                    m.seq_lens[:1],
                    self.kv_cache_spec.block_size,
                    width,
                )
        elif width <= block_table_tensor.shape[1]:
            fast_path_state_block_ids = block_table_tensor[:1, :width]
        else:
            return _miss("state_block_table_capacity")

        batch_size = int(m.num_actual_tokens)
        if (
            batch_size < 1
            or batch_size > self.decode_cudagraph_max_bs
            or query_len > self.decode_cudagraph_max_bs
            or self.spec_state_indices_tensor.shape[1] < width
        ):
            return _miss("capacity")

        spec_token_size = min(width, query_len)
        if spec_token_size > self.spec_token_indx.numel():
            return _miss("token_index_capacity")
        common_buffers = self._ddtree_fast_common_buffers
        if common_buffers is not None:
            spec_token_capacity = common_buffers.spec_token_indx.numel()
            token_index_initialized_size = common_buffers.token_index_initialized_size
        else:
            spec_token_capacity = self.spec_token_indx.numel()
            token_index_initialized_size = self._spec_token_indx_initialized_size
        if spec_token_size > spec_token_capacity:
            return _miss("token_index_capacity")
        if spec_token_size > token_index_initialized_size:
            start_idx = token_index_initialized_size
            token_index_buffer = (
                common_buffers.spec_token_indx
                if common_buffers is not None
                else self.spec_token_indx
            )
            token_index_buffer[start_idx:spec_token_size].copy_(
                torch.arange(
                    start_idx,
                    spec_token_size,
                    dtype=torch.int32,
                    device=query_start_loc.device,
                ),
                non_blocking=True,
            )
            if common_buffers is not None:
                common_buffers.token_index_initialized_size = spec_token_size
            else:
                self._spec_token_indx_initialized_size = spec_token_size

        static_key = (batch_size, query_len, width)
        tail_initialized = getattr(self, "_ddtree_fast_tail_key", None) == static_key
        common_tail_initialized = (
            common_buffers is not None and common_buffers.initialized_key == static_key
        )
        update_common = True
        if common_buffers is not None and fast_build_epoch is not None:
            update_common = (
                common_buffers.updated_epoch != fast_build_epoch
                or not common_tail_initialized
            )

        profile_graph_buffers_t0 = time.perf_counter() if metadata_profile else 0.0
        spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
        spec_sequence_masks_source = (
            common_buffers.spec_sequence_masks
            if common_buffers is not None
            else self.spec_sequence_masks
        )
        spec_token_indx_source = (
            common_buffers.spec_token_indx
            if common_buffers is not None
            else self.spec_token_indx
        )
        non_spec_token_indx_source = (
            common_buffers.non_spec_token_indx
            if common_buffers is not None
            else self.non_spec_token_indx
        )
        spec_query_start_loc_source = (
            common_buffers.spec_query_start_loc
            if common_buffers is not None
            else self.spec_query_start_loc
        )
        num_accepted_tokens_source = (
            common_buffers.num_accepted_tokens
            if common_buffers is not None
            else self.num_accepted_tokens
        )
        spec_state_slot_selectors_source = (
            common_buffers.spec_state_slot_selectors
            if common_buffers is not None
            else self.spec_state_slot_selectors
        )
        spec_sequence_masks = spec_sequence_masks_source[:batch_size]
        spec_token_indx = spec_token_indx_source[:spec_token_size]
        non_spec_token_indx = non_spec_token_indx_source[:0]
        spec_query_start_loc = spec_query_start_loc_source[: batch_size + 1]
        num_accepted_tokens_padded = num_accepted_tokens_source[:batch_size]

        selector_source = (
            num_accepted_tokens
            if spec_state_slot_selectors is None
            else spec_state_slot_selectors
        )
        if selector_source.ndim != 1 or selector_source.numel() < 1:
            return _miss("selector_shape")
        spec_state_slot_selectors_padded = spec_state_slot_selectors_source[:batch_size]
        use_triton_update = (
            _dflash_ddtree_gdn_fast_build_triton_enabled()
            and fast_path_state_block_ids.is_cuda
            and num_accepted_tokens.is_cuda
            and selector_source.is_cuda
            and spec_state_indices_tensor.is_contiguous()
            and fast_path_state_block_ids.stride(-1) == 1
        )
        if use_triton_update:
            state_block = (
                1 << max(batch_size * width, batch_size + 1, width).bit_length()
            )
            _ddtree_gdn_fast_metadata_kernel[(1,)](
                fast_path_state_block_ids,
                spec_state_indices_tensor,
                spec_sequence_masks,
                spec_query_start_loc,
                num_accepted_tokens_padded,
                spec_state_slot_selectors_padded,
                num_accepted_tokens,
                selector_source,
                width,
                batch_size,
                query_len,
                tail_initialized,
                update_common,
                common_tail_initialized
                if common_buffers is not None
                else tail_initialized,
                state_block,
            )
        else:
            spec_state_indices_tensor[:1].copy_(
                fast_path_state_block_ids, non_blocking=True
            )
            if not tail_initialized:
                spec_state_indices_tensor[1:].fill_(PAD_SLOT_ID)

            if update_common:
                spec_sequence_masks[:1].fill_(True)
                if not (
                    common_tail_initialized
                    if common_buffers is not None
                    else tail_initialized
                ):
                    spec_sequence_masks[1:].fill_(False)

                spec_query_start_loc[:1].fill_(0)
                if not (
                    common_tail_initialized
                    if common_buffers is not None
                    else tail_initialized
                ):
                    spec_query_start_loc[1:].fill_(query_len)

                num_accepted_tokens_padded[:1].copy_(
                    num_accepted_tokens[:1], non_blocking=True
                )
                if not (
                    common_tail_initialized
                    if common_buffers is not None
                    else tail_initialized
                ):
                    num_accepted_tokens_padded[1:].fill_(1)

                spec_state_slot_selectors_padded[:1].copy_(
                    selector_source[:1], non_blocking=True
                )
                if not (
                    common_tail_initialized
                    if common_buffers is not None
                    else tail_initialized
                ):
                    spec_state_slot_selectors_padded[1:].fill_(1)
        self._ddtree_fast_tail_key = static_key
        if common_buffers is not None and update_common:
            common_buffers.initialized_key = static_key
            common_buffers.updated_epoch = fast_build_epoch

        profile_graph_buffers_ms = 0.0
        if metadata_profile:
            profile_graph_buffers_ms = (
                time.perf_counter() - profile_graph_buffers_t0
            ) * 1000.0

        profile_register_t0 = time.perf_counter() if metadata_profile else 0.0
        metadata_key = (
            static_key,
            int(ddtree_parent_ids.data_ptr()),
            tuple(ddtree_parent_ids.shape),
            int(ddtree_num_tree_tokens_cpu.data_ptr()),
            tuple(ddtree_num_tree_tokens_cpu.shape),
        )
        cache_metadata = _dflash_ddtree_gdn_fast_build_cache_enabled()
        cached_metadata = None
        if (
            cache_metadata
            and getattr(self, "_ddtree_fast_metadata_key", None) == metadata_key
        ):
            cached_metadata = getattr(self, "_ddtree_fast_metadata", None)
        if cached_metadata is None:
            attn_metadata = GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                num_spec_decodes=1,
                num_spec_decode_tokens=query_len,
                num_actual_tokens=m.num_actual_tokens,
                has_initial_state=None,
                chunk_indices=None,
                chunk_offsets=None,
                spec_query_start_loc=spec_query_start_loc,
                non_spec_query_start_loc=None,
                spec_state_indices_tensor=spec_state_indices_tensor,
                non_spec_state_indices_tensor=None,
                spec_sequence_masks=spec_sequence_masks,
                spec_token_indx=spec_token_indx,
                non_spec_token_indx=non_spec_token_indx,
                num_accepted_tokens=num_accepted_tokens_padded,
                spec_state_slot_selectors=spec_state_slot_selectors_padded,
                ddtree_parent_ids=ddtree_parent_ids,
                ddtree_num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu,
                nums_dict=None,
                batch_ptr=None,
                token_chunk_offset_ptr=None,
            )
            register_gdn_spec_metadata_tensors(
                self.layer_names,
                gdn_spec_metadata_tensors(attn_metadata, query_start_loc.device),
            )
            if cache_metadata:
                self._ddtree_fast_metadata_key = metadata_key
                self._ddtree_fast_metadata = attn_metadata
        else:
            attn_metadata = cached_metadata
        profile_register_ms = 0.0
        if metadata_profile:
            profile_register_ms = (time.perf_counter() - profile_register_t0) * 1000.0
            logger.info(
                "DFLASH_DDTREE_METADATA_PROFILE gdn_build_fast total_ms=%.3f "
                "graph_buffers_ms=%.3f register_ms=%.3f tail_cached=%s "
                "common_cached=%s common_updated=%s cache_hit=%s "
                "num_spec_decodes=%d num_spec_decode_tokens=%d "
                "num_actual_tokens=%d full_graph=%s ddtree=%s layers=%d",
                (time.perf_counter() - metadata_profile_t0) * 1000.0,
                profile_graph_buffers_ms,
                profile_register_ms,
                tail_initialized,
                common_tail_initialized,
                update_common,
                cached_metadata is not None,
                1,
                query_len,
                m.num_actual_tokens,
                self.use_full_cuda_graph,
                True,
                len(self.layer_names),
            )

        return attn_metadata

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        spec_state_slot_selectors: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        spec_sequence_masks_cpu: torch.Tensor | None = None,
        current_state_block_ids: torch.Tensor | None = None,
        ddtree_parent_ids: torch.Tensor | None = None,
        ddtree_num_tree_tokens_cpu: torch.Tensor | None = None,
        ddtree_fast_build_epoch: int | None = None,
        for_cudagraph_capture: bool = False,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        metadata_profile = _dflash_ddtree_metadata_profile_enabled()
        metadata_profile_t0 = time.perf_counter() if metadata_profile else 0.0
        profile_state_contract_ms = 0.0
        profile_graph_buffers_ms = 0.0
        profile_register_ms = 0.0
        m = common_attn_metadata

        query_start_loc = m.query_start_loc
        query_start_loc_cpu = m.query_start_loc_cpu
        if fast_build and _dflash_ddtree_gdn_fast_build_enabled():
            fast_metadata = self._build_fast_pure_ddtree_full_graph(
                common_attn_metadata=common_attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                spec_state_slot_selectors=spec_state_slot_selectors,
                num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
                current_state_block_ids=current_state_block_ids,
                ddtree_parent_ids=ddtree_parent_ids,
                ddtree_num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu,
                for_cudagraph_capture=for_cudagraph_capture,
                fast_build_epoch=ddtree_fast_build_epoch,
                metadata_profile=metadata_profile,
                metadata_profile_t0=metadata_profile_t0,
            )
            if fast_metadata is not None:
                return fast_metadata
        self._ddtree_fast_tail_key = None
        _т0 = _time.perf_counter()
        # ОБЩИЙ КЛЮЧ ШАГА: у всех KV-групп GDN эти объекты -- одни и те же (различаются только
        # таблица блоков и слоты). Тождество объектов и есть признак «тот же шаг, та же группа дел».
        _клч0 = (id(m.query_start_loc), id(m.seq_lens), m.num_actual_tokens, m.num_reqs,
                 int(m.max_seq_len))  # см. пояснение в _общий_ключ: без длины ключ не различает ШАГИ
        _делить = _ДЕЛИТЬ_ОБЩЕЕ
        _общ0 = _ОБЩЕЕ.get("знач0") if (_делить and _ОБЩЕЕ.get("ключ0") == _клч0) else None
        if _общ0 is not None:
            context_lens_tensor = _общ0["ctx"]
        else:
            context_lens_tensor = m.compute_num_computed_tokens()
        _гф("0 compute_num_computed_tokens", _т0)
        nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
        mamba_cache_mode = self.vllm_config.cache_config.mamba_cache_mode
        mamba_all = mamba_cache_mode == "all"
        _т1 = _time.perf_counter()
        block_table_tensor = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            self.kv_cache_spec,
            mamba_cache_mode,
        )
        _гф("1 таблица блоков (режим)", _т1)
        # РЕЖИМ 'all' (задача 194). Здесь block_table_tensor приходит ПОЛНЫЙ (запрос x блоки),
        # и одномерные потребители (спекуляция, декод) его использовать не могут. Сведённую
        # таблицу берём ТОЙ ЖЕ функцией в режиме 'align': она отдаёт 1+num_spec последних
        # блоков каждого запроса -- ровно то, что нужно и спекуляции, и декоду. Собственного
        # второго выражения для этого НЕ пишем: расхождение двух тел уже стоило нам подъёма.
        aligned_block_table = None
        block_idx_last_computed_token = None
        aligned_block_table = None
        block_idx_last_scheduled_token = None
        block_idx_first_scheduled_token = None
        if mamba_all:
            _т2 = _time.perf_counter()
            aligned_block_table = mamba_get_block_table_tensor(
                m.block_table_tensor, m.seq_lens, self.kv_cache_spec, "align"
            )
            # [31.08 ГРАНИЦА БЛОКА СОСТОЯНИЯ] Таблица выше начинается с блока ПОСЛЕДНЕГО
            # ЗАПЛАНИРОВАННОГО токена: `(seq_lens-1)//block_size`. При спекуляции seq_lens
            # включает k+1 черновых, поэтому на шаге перехода через границу начало
            # перепрыгивает в НОВЫЙ блок, а состояние ещё лежит в старом -- блока со
            # состоянием в колонках просто нет. Замер прибором (FA2SM70_GDN_GRAN_DIAG):
            #     seq_len=4097 ctx=4093 start_po_seq=1 start_po_ctx=0
            #     seq_len=4098 ctx=4094 start_po_seq=1 start_po_ctx=0
            #     seq_len=4099 ctx=4095 start_po_seq=1 start_po_ctx=0
            #     seq_len=4100 ctx=4096 start_po_seq=1 start_po_ctx=0
            # (разность seq-ctx = 4 = k+1, то есть это ровно спекулятивный декод.)
            # Не-спекулятивный путь этого не знает: он берёт read по посчитанным, write по
            # запланированным, и потому границу проходит верно. Спекуляции даём ТУ ЖЕ опору --
            # начало по ПОСЧИТАННЫМ. Тогда колонка 0 снова указывает на блок с состоянием, а
            # колонки 1..num_spec покрывают блоки, куда k+1 токенов могут перейти.
            aligned_spec_table = aligned_block_table
            if _СПЕК_ПО_CTX:
                aligned_spec_table = mamba_get_block_table_tensor(
                    m.block_table_tensor, context_lens_tensor, self.kv_cache_spec, "align"
                )
            # [31.08 ДИАГНОСТИКА ГРАНИЦЫ] Печатает шаг, где начало align-таблицы, взятое по
            # seq_lens (последний ЗАПЛАНИРОВАННЫЙ токен), расходится с началом по числу
            # ПОСЧИТАННЫХ. Ровно в этот шаг спекуляция читает состояние из нового блока.
            if _ГРАН_ДИАГ and num_accepted_tokens is not None:
                # [04.09 СДВИГ ОПОРЫ] Колонки spec-таблицы -- это блоки, отсчитанные от
                # start=(seq_len-1)//B. Ядро пишет состояние позиции j в колонку j, а на
                # следующем шаге читает колонку (num_accepted-1). Значит согласованность
                # держится ТОЛЬКО пока start не сдвинулся между шагами. Прибор считает оба
                # старта: текущий и тот, что был при ЗАПИСИ (seq_prev-1 = ctx - m + k).
                try:
                    _B3 = self.kv_cache_spec.block_size
                    _m3 = num_accepted_tokens[: int(m.num_reqs)].to(torch.int64)
                    _ctx3 = context_lens_tensor[: int(m.num_reqs)].to(torch.int64)
                    _seq3 = m.seq_lens[: int(m.num_reqs)].to(torch.int64)
                    _st_cur = ((_seq3 - 1) // _B3).clamp(min=0)
                    _st_зап = ((_ctx3 - _m3 + self.num_spec) // _B3).clamp(min=0)
                    if _СЧЁТ.get("q", 0) < 3:
                        _СЧЁТ["q"] = _СЧЁТ.get("q", 0) + 1
                        print(f"[Q ШАГА] seq-ctx={int(_seq3[0]) - int(_ctx3[0])} "
                              f"ctx={int(_ctx3[0])} m={int(_m3[0])}",
                              file=_sys.stderr, flush=True)
                    _расх = (_st_cur != _st_зап)
                    if bool(_расх.any()):
                        _j = int(_расх.nonzero()[0][0])
                        _СЧЁТ["сдвиг"] = _СЧЁТ.get("сдвиг", 0) + 1
                        print(f"[СДВИГ ОПОРЫ] ctx={int(_ctx3[_j])} seq={int(_seq3[_j])} "
                              f"m={int(_m3[_j])} start_тек={int(_st_cur[_j])} "
                              f"start_зап={int(_st_зап[_j])} B={_B3} "
                              f"всего={_СЧЁТ['сдвиг']}", file=_sys.stderr, flush=True)
                except Exception as _e3:
                    print(f"[СДВИГ ОПОРЫ] err: {_e3}", file=_sys.stderr, flush=True)
            if _ГРАН_ДИАГ:
                try:
                    _bs2 = self.kv_cache_spec.block_size
                    _st_seq = ((m.seq_lens - 1) // _bs2).clamp(min=0)
                    _st_ctx = ((context_lens_tensor - 1) // _bs2).clamp(min=0)
                    _разн = (_st_seq != _st_ctx)
                    if bool(_разн.any()):
                        _i = int(_разн.nonzero()[0][0])
                        print(f"[ГРАНИЦА] zapros={_i} seq_len={int(m.seq_lens[_i])} "
                              f"ctx={int(context_lens_tensor[_i])} start_po_seq={int(_st_seq[_i])} "
                              f"start_po_ctx={int(_st_ctx[_i])} nreqs={int(m.num_reqs)}",
                              file=_sys.stderr, flush=True)
                except Exception as _e:
                    print(f"[ГРАНИЦА] diag err: {_e}", file=_sys.stderr, flush=True)
            _гф("2 таблица блоков (align)", _т2)
            mamba_block_size = self.kv_cache_spec.block_size
            if _общ0 is not None:
                # Индексы блоков зависят ТОЛЬКО от длин (контекст/последовательность), а они у
                # всех групп общие -- считаем один раз на шаг. Блоко-зависимое ниже своё.
                block_idx_last_computed_token = _общ0["b_last_computed"]
                block_idx_first_scheduled_token = _общ0["b_first_sched"]
                block_idx_last_scheduled_token = _общ0["b_last_sched"]
            else:
                # Индексы блоков -- как у Mamba2 (mamba_attn._compute_prefix_caching_block_indices)
                # [ТРИ ЛЕСЕНКИ В ОДНУ, 08.09] Те же три выражения считаются на СКЛЕЙКЕ
                # трёх рядов: одно деление с округлением вверх и одно вычитание вместо трёх.
                # Зажим остаётся РАЗНЫМ (у первого и третьего он есть, у второго нет) -- это
                # не косметика, а смысл: `first_scheduled` обязан уметь -1.
                _ряды = torch.stack((context_lens_tensor,
                                     context_lens_tensor + 1,
                                     m.seq_lens))
                _ряды = cdiv(_ряды, mamba_block_size) - 1
                block_idx_last_computed_token = _ряды[0].clamp(min=0)
                block_idx_first_scheduled_token = _ряды[1]
                block_idx_last_scheduled_token = _ряды[2].clamp(min=0)
                if _делить:
                    _ОБЩЕЕ["ключ0"] = _клч0
                    _ОБЩЕЕ["якорь0"] = (m.query_start_loc, m.seq_lens)  # см. _общий_ключ
                    _ОБЩЕЕ["знач0"] = {
                        "ctx": context_lens_tensor,
                        "b_last_computed": block_idx_last_computed_token,
                        "b_first_sched": block_idx_first_scheduled_token,
                        "b_last_sched": block_idx_last_scheduled_token,
                    }
                    _общ0 = _ОБЩЕЕ["знач0"]
            _гф("3 индексы блоков", _т2)
        is_mamba_cache_all = mamba_all

        num_reqs = query_start_loc_cpu.numel() - 1
        if spec_sequence_masks_cpu is not None:
            assert spec_sequence_masks_cpu.dtype == torch.bool
            assert spec_sequence_masks_cpu.ndim == 1
            assert spec_sequence_masks_cpu.numel() == num_reqs, (
                f"spec_sequence_masks_cpu.shape={tuple(spec_sequence_masks_cpu.shape)} "
                f"must align with num_reqs={num_reqs}"
            )
        if num_decode_draft_tokens_cpu is not None:
            assert num_decode_draft_tokens_cpu.ndim == 1
            assert num_decode_draft_tokens_cpu.numel() == num_reqs, (
                "num_decode_draft_tokens_cpu must align with query_start_loc"
            )
        if num_accepted_tokens is not None:
            assert num_accepted_tokens.ndim == 1
            assert num_accepted_tokens.numel() == num_reqs, (
                "num_accepted_tokens must align with query_start_loc"
            )
        if spec_state_slot_selectors is not None:
            assert spec_state_slot_selectors.ndim == 1
            assert spec_state_slot_selectors.numel() == num_reqs, (
                "spec_state_slot_selectors must align with query_start_loc"
            )
        if ddtree_parent_ids is not None:
            assert ddtree_parent_ids.ndim == 2
            assert ddtree_parent_ids.shape[0] == num_reqs, (
                "ddtree_parent_ids must align with query_start_loc"
            )
            assert ddtree_num_tree_tokens_cpu is not None
            assert ddtree_num_tree_tokens_cpu.ndim == 1
            assert ddtree_num_tree_tokens_cpu.numel() == num_reqs, (
                "ddtree_num_tree_tokens_cpu must align with query_start_loc"
            )

        _тА = _time.perf_counter()
        if not self.use_spec_decode:
            spec_sequence_masks = None
            num_spec_decodes = 0
        else:
            if spec_sequence_masks_cpu is None:
                if num_decode_draft_tokens_cpu is None:
                    spec_sequence_masks = None
                    num_spec_decodes = 0
                    spec_sequence_masks_cpu = None
                else:
                    spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
            if (
                spec_sequence_masks_cpu is None
                or spec_sequence_masks_cpu.sum().item() == 0
            ):
                spec_sequence_masks = None
                num_spec_decodes = 0
                spec_sequence_masks_cpu = None
            else:
                if num_decode_draft_tokens_cpu is not None:
                    num_spec_draft_tokens = (
                        num_decode_draft_tokens_cpu[spec_sequence_masks_cpu]
                        .sum()
                        .item()
                    )
                    if num_spec_draft_tokens == 0:
                        spec_sequence_masks = None
                        num_spec_decodes = 0
                        spec_sequence_masks_cpu = None
                    else:
                        num_spec_decodes = spec_sequence_masks_cpu.sum().item()
                        spec_sequence_masks = spec_sequence_masks_cpu.to(
                            query_start_loc.device, non_blocking=_НЕБЛОК
                        )
                else:
                    num_spec_decodes = spec_sequence_masks_cpu.sum().item()
                    spec_sequence_masks = spec_sequence_masks_cpu.to(
                        query_start_loc.device, non_blocking=_НЕБЛОК
                    )
        # ПРЕФИКСНОСТЬ маски -- по ПРОЦЕССОРНОЙ копии: без обращения к карте и без
        # синхронизации. `_преф_спек` = число ведущих истин, если маска ровно префиксная.
        _преф_спек = None
        if _ОДНОРОДН and spec_sequence_masks_cpu is not None:
            _к = int(spec_sequence_masks_cpu.sum())
            if bool(spec_sequence_masks_cpu[:_к].all()) and not bool(
                    spec_sequence_masks_cpu[_к:].any()):
                _преф_спек = _к

        _гф("A маски", _тА)
        _тБ = _time.perf_counter()
        if spec_sequence_masks is None:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
            num_spec_decode_tokens = 0
            spec_token_indx = None
            non_spec_token_indx = None
            spec_state_indices_tensor = None
            if is_mamba_cache_all:
                # [FA2/SM70, задача 194] В 'all' слой получает ВСЮ таблицу блоков (он сам
                # возьмёт нужный столбец по block_idx_*); в 'none'/'align' таблица уже сведена
                # к одному блоку на запрос. [ПОРТ 07.2026] Новый upstream здесь сводил таблицу
                # к одному столбцу (gather по seq_lens) -- наш слой (qwen3_next, признак
                # mamba_block_size > 0) требует двумерную таблицу, поэтому оставлена наша форма.
                non_spec_state_indices_tensor = block_table_tensor
            else:
                non_spec_state_indices_tensor = select_gdn_state_block_ids(
                    block_table_tensor,
                    num_accepted_tokens,
                    self.num_spec_state_tokens,
                )
            if num_prefills == 0:
                query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
                if torch.any(query_lens_cpu == 0):
                    decode_lane_mask = (query_lens_cpu > 0).to(
                        device=non_spec_state_indices_tensor.device,
                        non_blocking=True,
                    )
                    if non_spec_state_indices_tensor.dim() == 2:
                        # [ПОРТ 07.2026] Двумерная таблица режима 'all': маска -- по строкам.
                        decode_lane_mask = decode_lane_mask.unsqueeze(1)
                    non_spec_state_indices_tensor = torch.where(
                        decode_lane_mask,
                        non_spec_state_indices_tensor,
                        torch.full_like(non_spec_state_indices_tensor, PAD_SLOT_ID),
                    )
            if for_cudagraph_capture and num_prefills == 0:
                # Capture/dummy runs must not mutate real recurrent state.
                # State slot 0 is live, so padding has to use PAD_SLOT_ID.
                non_spec_state_indices_tensor = torch.full_like(
                    non_spec_state_indices_tensor, PAD_SLOT_ID
                )
            spec_query_start_loc = None
            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
            if (
                self.use_full_cuda_graph
                and self.use_spec_decode
                and envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP
            ):
                placeholder_rows = min(
                    self.num_spec_state_tokens + 1,
                    self.decode_cudagraph_max_bs,
                )
                self.spec_state_indices_tensor[:placeholder_rows].fill_(PAD_SLOT_ID)
                spec_state_indices_tensor = self.spec_state_indices_tensor[
                    :placeholder_rows
                ]
                self.spec_sequence_masks[:placeholder_rows].fill_(False)
                spec_sequence_masks = self.spec_sequence_masks[:placeholder_rows]
                self.spec_query_start_loc[: placeholder_rows + 1].fill_(0)
                spec_query_start_loc = self.spec_query_start_loc[: placeholder_rows + 1]
                self.spec_token_indx[:placeholder_rows].copy_(
                    torch.arange(
                        placeholder_rows,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    ),
                    non_blocking=True,
                )
                spec_token_indx = self.spec_token_indx[:placeholder_rows]
                self.num_accepted_tokens[:placeholder_rows].fill_(1)
                num_accepted_tokens = self.num_accepted_tokens[:placeholder_rows]
        else:
            assert spec_sequence_masks_cpu is not None
            assert num_accepted_tokens is not None
            query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            non_spec_query_lens_cpu = query_lens_cpu[~spec_sequence_masks_cpu]
            num_zero_len = (non_spec_query_lens_cpu == 0).sum().item()
            pure_ddtree_spec_fast_path_candidate = (
                ddtree_parent_ids is not None
                and num_spec_decodes > 0
                and non_spec_query_lens_cpu.size(0) == num_zero_len
            )
            fast_path_state_block_ids: torch.Tensor | None = None
            if pure_ddtree_spec_fast_path_candidate:
                if current_state_block_ids is not None:
                    fast_path_state_block_ids = current_state_block_ids[
                        :num_spec_decodes, : self.num_spec_state_tokens + 1
                    ]
                elif bool(torch.all(spec_sequence_masks_cpu).item()):
                    fast_path_width = self.num_spec_state_tokens + 1
                    if is_mamba_cache_all:
                        seq_lens_cpu_for_state = m._seq_lens_cpu
                        if seq_lens_cpu_for_state is not None and num_spec_decodes == 1:
                            seq_len = int(seq_lens_cpu_for_state[0].item())
                            start_idx = max(
                                (seq_len - 1) // self.kv_cache_spec.block_size,
                                0,
                            )
                            end_idx = start_idx + fast_path_width
                            if end_idx <= block_table_tensor.shape[1]:
                                fast_path_state_block_ids = block_table_tensor[
                                    :1, start_idx:end_idx
                                ]
                        else:
                            fast_path_state_block_ids = gather_gdn_state_block_ids(
                                block_table_tensor[:num_spec_decodes],
                                m.seq_lens[:num_spec_decodes],
                                self.kv_cache_spec.block_size,
                                fast_path_width,
                            )
                    elif fast_path_width <= block_table_tensor.shape[1]:
                        fast_path_state_block_ids = block_table_tensor[
                            :num_spec_decodes, :fast_path_width
                        ]

            pure_ddtree_spec_fast_path = fast_path_state_block_ids is not None
            if pure_ddtree_spec_fast_path:
                profile_state_contract_t0 = (
                    time.perf_counter() if metadata_profile else 0.0
                )
                num_decodes = 0
                num_prefills = 0
                num_decode_tokens = 0
                num_prefill_tokens = 0
                num_spec_decode_tokens = query_lens_cpu.sum().item()
                spec_token_size = min(
                    num_spec_decodes * (self.num_spec_state_tokens + 1),
                    query_start_loc_cpu[-1].item(),
                )
                if spec_token_size <= self.spec_token_indx.numel():
                    if spec_token_size > self._spec_token_indx_initialized_size:
                        start_idx = self._spec_token_indx_initialized_size
                        self.spec_token_indx[start_idx:spec_token_size].copy_(
                            torch.arange(
                                start_idx,
                                spec_token_size,
                                dtype=torch.int32,
                                device=query_start_loc.device,
                            ),
                            non_blocking=True,
                        )
                        self._spec_token_indx_initialized_size = spec_token_size
                    spec_token_indx = self.spec_token_indx[:spec_token_size]
                else:
                    spec_token_indx = torch.arange(
                        spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                non_spec_token_indx = self.non_spec_token_indx[:0]
                spec_state_indices_tensor = fast_path_state_block_ids
                if for_cudagraph_capture:
                    spec_state_indices_tensor = torch.full_like(
                        spec_state_indices_tensor, PAD_SLOT_ID
                    )
                non_spec_state_indices_tensor = None
                spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                non_spec_query_start_loc = None
                non_spec_query_start_loc_cpu = None
                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]
                if spec_state_slot_selectors is None:
                    spec_state_slot_selectors = num_accepted_tokens
                else:
                    spec_state_slot_selectors = spec_state_slot_selectors[
                        :num_spec_decodes
                    ]
                if metadata_profile:
                    profile_state_contract_ms = (
                        time.perf_counter() - profile_state_contract_t0
                    ) * 1000.0
            else:
                # query_start_loc may be padded for CUDA graph replay. The CPU
                # metadata is authoritative for the live request count here.
                query_lens = query_lens_cpu.to(
                    query_start_loc.device, non_blocking=True
                )
                profile_state_contract_t0 = (
                    time.perf_counter() if metadata_profile else 0.0
                )
                state_contract = build_gdn_spec_decode_state_contract(
                    block_table_tensor=block_table_tensor,
                    seq_lens=m.seq_lens,
                    block_size=self.kv_cache_spec.block_size,
                    num_spec=self.num_spec_state_tokens,
                    spec_sequence_masks_cpu=spec_sequence_masks_cpu,
                    num_accepted_tokens=num_accepted_tokens,
                    current_state_block_ids=current_state_block_ids,
                    is_mamba_cache_all=is_mamba_cache_all,
                    spec_state_slot_selectors=spec_state_slot_selectors,
                    _преф_спек=_преф_спек,
                )
                if mamba_all:
                    # [FA2/SM70, задача 194] Спекуляция всегда работает СВЕДЁННОЙ таблицей: её
                    # токены идут за последним запланированным, и им нужны 1+num_spec
                    # ПОСЛЕДНИХ блоков -- ровно то, что отдаёт режим 'align' (опционально по
                    # контексту, FA2SM70_GDN_SPEC_CTX). Не-спекулятивные строки в 'all' получают
                    # ВСЮ таблицу блоков (двумерно): наш слой берёт столбец сам по block_idx_*.
                    # [ПОРТ 07.2026] Новый upstream в 'all' сводил не-спекулятивные строки к
                    # одному столбцу (gather по seq_lens) -- заменено нашей формой.
                    state_contract.spec_state_indices_tensor = _выбор(
                        aligned_spec_table, spec_sequence_masks, _преф_спек, True
                    )[:, : self.num_spec_state_tokens + 1]
                    state_contract.non_spec_state_indices_tensor = _выбор(
                        block_table_tensor, spec_sequence_masks, _преф_спек, False
                    )
                if metadata_profile:
                    profile_state_contract_ms = (
                        time.perf_counter() - profile_state_contract_t0
                    ) * 1000.0

                # [FA2/SM70 25.08] ОБЩИЙ СЧЁТ ГРУПП: счётчики, индексы токенов и начала
                # запросов у всех KV-групп GDN совпадают -- считаем один раз на шаг.
                _клч = _общий_ключ(m, num_accepted_tokens)
                _общ = _ОБЩЕЕ["знач"] if _ОБЩЕЕ["ключ"] == _клч else None
                if _общ is not None and _ДЕЛИТЬ_ОБЩЕЕ:
                    (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens,
                     num_spec_decode_tokens, spec_token_indx, non_spec_token_indx,
                     spec_query_start_loc, non_spec_query_start_loc,
                     non_spec_query_start_loc_cpu) = _общ
                    # БЛОКО-ЗАВИСИМОЕ -- СВОЁ У КАЖДОЙ ГРУППЫ.
                    spec_state_indices_tensor = state_contract.spec_state_indices_tensor
                    non_spec_state_indices_tensor = (
                        None
                        if num_prefills == 0 and num_decodes == 0
                        else state_contract.non_spec_state_indices_tensor
                    )
                    if for_cudagraph_capture:
                        spec_state_indices_tensor = torch.full_like(
                            spec_state_indices_tensor, PAD_SLOT_ID
                        )
                        if non_spec_state_indices_tensor is not None:
                            non_spec_state_indices_tensor = torch.full_like(
                                non_spec_state_indices_tensor, PAD_SLOT_ID
                            )
                else:
                    if envs.VLLM_SM70_MTP_LEGACY_GDN_MIXED_DECODE_ROUTING:
                        # 0.0.3 kept ordinary query_len==1 rows on the decode path even
                        # when another row was running speculative verification. This
                        # is an A/B guard for MTP-only recurrent-state corruption.
                        num_decodes = (non_spec_query_lens_cpu == 1).sum().item()
                        num_prefills = (
                            non_spec_query_lens_cpu.size(0) - num_decodes - num_zero_len
                        )
                        num_decode_tokens = num_decodes
                        num_prefill_tokens = (
                            non_spec_query_lens_cpu.sum().item() - num_decode_tokens
                        )
                    else:
                        # When active spec decodes are present, route non-spec requests
                        # through the prefill path so mixed batches keep separate GDN
                        # state metadata for spec and non-spec tokens.
                        num_decodes = 0
                        num_prefills = non_spec_query_lens_cpu.size(0) - num_zero_len
                        num_decode_tokens = 0
                        num_prefill_tokens = non_spec_query_lens_cpu.sum().item()
                    num_spec_decode_tokens = (
                        query_lens_cpu.sum().item() - num_prefill_tokens - num_decode_tokens
                    )

                    if num_prefills == 0 and num_decodes == 0:
                        spec_token_size = min(
                            num_spec_decodes * (self.num_spec_state_tokens + 1),
                            query_start_loc_cpu[-1].item(),
                        )
                        spec_token_indx = torch.arange(
                            spec_token_size,
                            dtype=torch.int32,
                            device=query_start_loc.device,
                        )
                        non_spec_token_indx = torch.empty(
                            0, dtype=torch.int32, device=query_start_loc.device
                        )
                        spec_state_indices_tensor = state_contract.spec_state_indices_tensor
                        if for_cudagraph_capture:
                            spec_state_indices_tensor = torch.full_like(
                                spec_state_indices_tensor, PAD_SLOT_ID
                            )
                        non_spec_state_indices_tensor = None
                        # Padded sequences are always at the back, so the first
                        # num_spec_decodes + 1 entries of query_start_loc already
                        # contain the correct cumulative token counts.
                        spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                        non_spec_query_start_loc = None
                        non_spec_query_start_loc_cpu = None
                    else:
                        spec_token_masks = torch.repeat_interleave(
                            spec_sequence_masks,
                            query_lens,
                            output_size=query_start_loc_cpu[-1].item(),
                        )
                        index = torch.argsort(spec_token_masks, stable=True)
                        num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
                        non_spec_token_indx = index[:num_non_spec_tokens]
                        spec_token_indx = index[num_non_spec_tokens:]

                        spec_state_indices_tensor = state_contract.spec_state_indices_tensor
                        non_spec_state_indices_tensor = (
                            state_contract.non_spec_state_indices_tensor
                        )
                        if for_cudagraph_capture:
                            spec_state_indices_tensor = torch.full_like(
                                spec_state_indices_tensor, PAD_SLOT_ID
                            )
                            non_spec_state_indices_tensor = torch.full_like(
                                non_spec_state_indices_tensor, PAD_SLOT_ID
                            )

                        spec_query_start_loc = torch.zeros(
                            num_spec_decodes + 1,
                            dtype=torch.int32,
                            device=query_start_loc.device,
                        )
                        torch.cumsum(
                            _выбор(query_lens, spec_sequence_masks, _преф_спек, True),
                            dim=0,
                            out=spec_query_start_loc[1:],
                        )
                        non_spec_query_start_loc = torch.zeros(
                            query_lens.size(0) - num_spec_decodes + 1,
                            dtype=torch.int32,
                            device=query_start_loc.device,
                        )
                        torch.cumsum(
                            _выбор(query_lens, spec_sequence_masks, _преф_спек, False),
                            dim=0,
                            out=non_spec_query_start_loc[1:],
                        )
                        non_spec_query_start_loc_cpu = torch.zeros(
                            query_lens_cpu.size(0) - num_spec_decodes + 1,
                            dtype=torch.int32,
                            device="cpu",
                        )
                        torch.cumsum(
                            query_lens_cpu[~spec_sequence_masks_cpu],
                            dim=0,
                            out=non_spec_query_start_loc_cpu[1:],
                        )

                    if _общ is None:
                        _ОБЩЕЕ["ключ"] = _клч
                        _ОБЩЕЕ["якорь"] = (m.query_start_loc, m.seq_lens, num_accepted_tokens)
                        _ОБЩЕЕ["знач"] = (
                            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens,
                            num_spec_decode_tokens, spec_token_indx, non_spec_token_indx,
                            spec_query_start_loc, non_spec_query_start_loc,
                            non_spec_query_start_loc_cpu,
                        )

                num_accepted_tokens = state_contract.num_accepted_tokens
                spec_state_slot_selectors = state_contract.spec_state_slot_selectors
            assert spec_query_start_loc is not None
            assert spec_query_start_loc[-1].item() == num_spec_decode_tokens
            assert spec_state_indices_tensor is not None
            assert spec_state_indices_tensor.shape[0] == num_spec_decodes

        chunk_indices: torch.Tensor | None = None
        chunk_offsets: torch.Tensor | None = None
        flashqla_original_prefill = (
            self.gdn_prefill_backend == "flashqla_sm70"
            and _sm70_flashqla_original_prefill_enabled()
        )
        if num_prefills > 0 and (
            self.gdn_prefill_backend != "flashqla_sm70" or flashqla_original_prefill
        ):
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            if self.gdn_prefill_backend == "cutedsl":
                from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                    prepare_metadata_cutedsl,
                )

                assert non_spec_query_start_loc is not None
                assert non_spec_query_start_loc_cpu is not None
                total_tokens = int(non_spec_query_start_loc_cpu[-1].item())
                chunk_indices, chunk_offsets = prepare_metadata_cutedsl(
                    non_spec_query_start_loc,
                    total_tokens,
                    FLA_CHUNK_SIZE,
                )
            else:
                gpu_device = query_start_loc.device
                # Only prefill batches use FLA chunk ops.
                # Pre-compute on CPU and async-copy to GPU to avoid
                # GPU→CPU sync (.tolist()) in prepare_chunk_indices.
                from vllm.model_executor.layers.fla.ops.index import (
                    prepare_chunk_indices,
                    prepare_chunk_offsets,
                )

                assert non_spec_query_start_loc_cpu is not None
                chunk_indices = prepare_chunk_indices(
                    non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE
                ).to(device=gpu_device, non_blocking=True)
                chunk_offsets = prepare_chunk_offsets(
                    non_spec_query_start_loc_cpu, FLA_CHUNK_SIZE
                ).to(device=gpu_device, non_blocking=True)

        if num_prefills > 0:
            has_initial_state = context_lens_tensor > 0
            if spec_sequence_masks_cpu is not None:
                has_initial_state = _выбор(has_initial_state, spec_sequence_masks_cpu,
                                           _преф_спек, False)
                assert non_spec_query_start_loc_cpu is not None
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(
                    non_spec_query_start_loc_cpu,
                    device=query_start_loc.device,
                )
            )
        else:
            has_initial_state = None

        # Prepare tensors for cudagraph
        # Note: m.num_actual_tokens is already padded by the model runner for CUDAGraph
        batch_size = m.num_actual_tokens

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_decodes == 0
            and num_spec_decodes <= self.decode_cudagraph_max_bs
            and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
        ):
            profile_graph_buffers_t0 = time.perf_counter() if metadata_profile else 0.0
            common_buffers = self._ddtree_fast_common_buffers
            spec_sequence_masks_buffer = (
                common_buffers.spec_sequence_masks
                if common_buffers is not None
                else self.spec_sequence_masks
            )
            spec_token_indx_buffer = (
                common_buffers.spec_token_indx
                if common_buffers is not None
                else self.spec_token_indx
            )
            non_spec_token_indx_buffer = (
                common_buffers.non_spec_token_indx
                if common_buffers is not None
                else self.non_spec_token_indx
            )
            spec_query_start_loc_buffer = (
                common_buffers.spec_query_start_loc
                if common_buffers is not None
                else self.spec_query_start_loc
            )
            num_accepted_tokens_buffer = (
                common_buffers.num_accepted_tokens
                if common_buffers is not None
                else self.num_accepted_tokens
            )
            spec_state_slot_selectors_buffer = (
                common_buffers.spec_state_slot_selectors
                if common_buffers is not None
                else self.spec_state_slot_selectors
            )
            assert spec_sequence_masks is not None
            self.spec_state_indices_tensor[:num_spec_decodes].copy_(
                spec_state_indices_tensor, non_blocking=_НЕБЛОК
            )
            spec_state_indices_tensor = self.spec_state_indices_tensor[:batch_size]
            spec_state_indices_tensor[num_spec_decodes:].fill_(PAD_SLOT_ID)

            spec_sequence_masks_buffer[:num_spec_decodes].copy_(
                spec_sequence_masks[:num_spec_decodes], non_blocking=_НЕБЛОК
            )
            spec_sequence_masks = spec_sequence_masks_buffer[:batch_size]
            spec_sequence_masks[num_spec_decodes:].fill_(False)

            assert non_spec_token_indx is not None and spec_token_indx is not None
            if non_spec_token_indx.numel() > 0:
                non_spec_token_indx_buffer[: non_spec_token_indx.size(0)].copy_(
                    non_spec_token_indx, non_blocking=_НЕБЛОК
                )
            _хв_нс = min(int(non_spec_token_indx_buffer.shape[0]),
                         int(self.decode_cudagraph_max_bs)
                         * (self.num_spec_state_tokens + 1))
            if int(non_spec_token_indx.size(0)) < _хв_нс:
                non_spec_token_indx_buffer[int(non_spec_token_indx.size(0)):_хв_нс].fill_(0)
            non_spec_token_indx = non_spec_token_indx_buffer[
                : non_spec_token_indx.size(0)
            ]

            if (
                spec_token_indx.numel() > 0
                and spec_token_indx.data_ptr() != spec_token_indx_buffer.data_ptr()
            ):
                spec_token_indx_buffer[: spec_token_indx.size(0)].copy_(
                    spec_token_indx, non_blocking=_НЕБЛОК
                )
            # [FA2/SM70] ХВОСТ ИНДЕКСОВ ТОКЕНОВ ОБНУЛЯЕТСЯ, И ЭТО НЕ ПЕДАНТИЗМ.
            # Полный граф захватывает ДЛИНУ этого среза; на воспроизведении настоящих
            # строк меньше, а ядро идёт по захваченной длине и читает хвост -- там лежат
            # индексы ПРОШЛОГО шага, и они могут указывать ЗА нынешнее число токенов.
            # Дальше это `index_select`/gather за пределами -- то есть Xid 13 и смерть
            # воркера. Все соседние спекулятивные буферы паддинг получают
            # (spec_state_indices, spec_sequence_masks, spec_query_start_loc,
            # num_accepted_tokens), а этот -- нет. Ноль всегда годный индекс.
            _хв_сп = min(int(spec_token_indx_buffer.shape[0]),
                         int(self.decode_cudagraph_max_bs)
                         * (self.num_spec_state_tokens + 1))
            _хвост_обнулён = int(spec_token_indx.size(0)) < _хв_сп
            if _хвост_обнулён:
                spec_token_indx_buffer[int(spec_token_indx.size(0)):_хв_сп].fill_(0)
            if common_buffers is not None:
                common_buffers.token_index_initialized_size = max(
                    common_buffers.token_index_initialized_size,
                    spec_token_indx.size(0),
                )
            # [ПОРТ 07.2026] Хвост за spec_token_indx.size(0) обнулён -- значит «уже
            # заполнено arange» (инвариант быстрого пути DDTree) верно лишь до этой длины.
            if _хвост_обнулён:
                if common_buffers is not None:
                    common_buffers.token_index_initialized_size = int(
                        spec_token_indx.size(0))
                if spec_token_indx_buffer is self.spec_token_indx:
                    self._spec_token_indx_initialized_size = min(
                        self._spec_token_indx_initialized_size,
                        int(spec_token_indx.size(0)))
            spec_token_indx = spec_token_indx_buffer[: spec_token_indx.size(0)]

            spec_query_start_loc_buffer[: num_spec_decodes + 1].copy_(
                spec_query_start_loc, non_blocking=_НЕБЛОК
            )
            spec_num_query_tokens = spec_query_start_loc[-1]  # type: ignore[index]
            spec_query_start_loc = spec_query_start_loc_buffer[: batch_size + 1]
            spec_query_start_loc[num_spec_decodes + 1 :].fill_(spec_num_query_tokens)

            num_accepted_tokens_buffer[:num_spec_decodes].copy_(
                num_accepted_tokens, non_blocking=_НЕБЛОК
            )
            num_accepted_tokens = num_accepted_tokens_buffer[:batch_size]
            num_accepted_tokens[num_spec_decodes:].fill_(1)

            spec_state_slot_selectors_buffer[:num_spec_decodes].copy_(
                spec_state_slot_selectors, non_blocking=_НЕБЛОК
            )
            spec_state_slot_selectors = spec_state_slot_selectors_buffer[:batch_size]
            spec_state_slot_selectors[num_spec_decodes:].fill_(1)
            if common_buffers is not None:
                common_buffers.initialized_key = (
                    batch_size,
                    int(num_spec_decode_tokens),
                    self.num_spec_state_tokens + 1,
                )
            if metadata_profile:
                profile_graph_buffers_ms = (
                    time.perf_counter() - profile_graph_buffers_t0
                ) * 1000.0

        if (
            self.use_full_cuda_graph
            and num_prefills == 0
            and num_spec_decodes == 0
            and num_decodes <= self.decode_cudagraph_max_bs
        ):
            self.non_spec_state_indices_tensor[:num_decodes].copy_(
                non_spec_state_indices_tensor,
                non_blocking=_НЕБЛОК,
            )
            non_spec_state_indices_tensor = self.non_spec_state_indices_tensor[
                :batch_size
            ]
            non_spec_state_indices_tensor[num_decodes:].fill_(PAD_SLOT_ID)

            if mamba_all:
                # Те же указатели блоков -- в постоянные буферы, по той же причине.
                self.block_idx_last_computed_token[:num_decodes].copy_(
                    block_idx_last_computed_token, non_blocking=_НЕБЛОК
                )
                block_idx_last_computed_token = self.block_idx_last_computed_token[
                    :batch_size
                ]
                block_idx_last_computed_token[num_decodes:].fill_(0)
                self.block_idx_last_scheduled_token[:num_decodes].copy_(
                    block_idx_last_scheduled_token, non_blocking=_НЕБЛОК
                )
                block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
                    :batch_size
                ]
                block_idx_last_scheduled_token[num_decodes:].fill_(0)

            self.non_spec_query_start_loc[: num_decodes + 1].copy_(
                non_spec_query_start_loc, non_blocking=_НЕБЛОК
            )
            non_spec_num_query_tokens = non_spec_query_start_loc[-1]  # type: ignore[index]
            non_spec_query_start_loc = self.non_spec_query_start_loc[: batch_size + 1]
            non_spec_query_start_loc[num_decodes + 1 :].fill_(non_spec_num_query_tokens)

        # РЕЖИМ 'all': индексы блоков приводим к тому же ПОДМНОЖЕСТВУ и тому же ПОРЯДКУ, что
        # и non_spec_state_indices_tensor (сперва декоды, затем префиллы), иначе состояние
        # запроса будет прочитано из блока соседа -- отказа не будет, будет тихая порча.
        _гф("Б ветка спекуляции + копии", _тБ)
        _тВ = _time.perf_counter()
        num_computed_tokens_ns = None
        num_computed_tokens_ns_cpu = None
        if mamba_all:
            # [ПОРТ 07.2026] Признак спекулятивного шага -- num_spec_decodes > 0: новый upstream
            # (VLLM_SM70_QWEN_GDN_SPEC_CORE_OP) отдаёт маску-заполнитель и на шаге БЕЗ спекуляции.
            if num_spec_decodes > 0 and spec_sequence_masks is not None:
                # ДЛИНЫ СОГЛАСУЕМ ЯВНО. При ЗАХВАТЕ ГРАФА метаданные приходят ПАДДИРОВАННЫМИ
                # (движок сам это признаёт: `unpadded()` в backend.py с пометкой «drafter still
                # only uses piecewise cudagraphs ... does not want padded metadata»), поэтому
                # маска бывает ДЛИННЕЕ индексов блоков: замерено отказом 22.08 -- «mask [8] does
                # not match indexed tensor [2]», подъём падал целиком. Паддинг всегда в ХВОСТЕ,
                # реальные запросы идут первыми, поэтому срез по длине индексов берёт ровно
                # реальную часть. Если длины совпали -- поведение прежнее.
                if _общ0 is not None and "b_last_computed_m" in _общ0:
                    block_idx_last_computed_token = _общ0["b_last_computed_m"]
                    block_idx_last_scheduled_token = _общ0["b_last_sched_m"]
                    block_idx_first_scheduled_token = _общ0["b_first_sched_m"]
                    num_computed_tokens_ns = _общ0["ns"]
                else:
                    _n = block_idx_last_computed_token.shape[0]
                    _keep = ~(spec_sequence_masks[:_n] if spec_sequence_masks.shape[0] > _n
                              else spec_sequence_masks)
                    # При однородной маске `_keep` сплошь ложна -- выборка даёт пустое, и
                    # `nonzero` внутри индексации не нужен вовсе (см. _выбор).
                    if _преф_спек is not None:
                        # [МИНА 12.09, боевой лёг 17:01] Префиксная маска НЕ значит «все строки
                        # спекулятивные»: за префиксом идут НЕ-спекулятивные строки -- в том числе
                        # ПРЕФИЛЛ соседнего запроса в том же батче. Прежде здесь брали `[:0]`, и при
                        # num_prefills > 0 цикл по qsl в qwen3_next читал пустой ncomp -> IndexError,
                        # движок умирал (8 клиентов получили 500). Берём строки ПОСЛЕ префикса --
                        # ровно то, что делает _выбор(..., брать_спек=False), без nonzero и синхронизации.
                        block_idx_last_computed_token = block_idx_last_computed_token[_преф_спек:]
                        block_idx_last_scheduled_token = block_idx_last_scheduled_token[_преф_спек:]
                        block_idx_first_scheduled_token = block_idx_first_scheduled_token[_преф_спек:]
                        num_computed_tokens_ns = context_lens_tensor[_преф_спек:]
                    else:
                        block_idx_last_computed_token = block_idx_last_computed_token[_keep]
                        block_idx_last_scheduled_token = block_idx_last_scheduled_token[_keep]
                        block_idx_first_scheduled_token = block_idx_first_scheduled_token[_keep]
                        num_computed_tokens_ns = context_lens_tensor[_keep]
                    if _делить and _общ0 is not None:
                        _общ0["b_last_computed_m"] = block_idx_last_computed_token
                        _общ0["b_last_sched_m"] = block_idx_last_scheduled_token
                        _общ0["b_first_sched_m"] = block_idx_first_scheduled_token
                        _общ0["ns"] = num_computed_tokens_ns
            else:
                num_computed_tokens_ns = context_lens_tensor
            if num_prefills > 0:
                # Синхронизация ТОЛЬКО когда в батче есть префилл (см. поле выше).
                num_computed_tokens_ns_cpu = num_computed_tokens_ns.to("cpu")

        _гф("В индексы блоков (mamba all)", _тВ)
        _тГ = _time.perf_counter()
        # [ГРАНИЦА БЛОКА СОСТОЯНИЯ -- ПОДСТАНОВКА БЛОКА-ИСТОЧНИКА, 04.09] ---------------
        # Колонки spec-таблицы -- это блоки, отсчитанные от опоры A=(seq_len-1)//B: ядро пишет
        # состояние позиции j в колонку j, а на СЛЕДУЮЩЕМ шаге читает колонку num_accepted-1.
        # Согласовано это лишь пока опора не двигалась. Опора же считается от seq_len, а он
        # растёт: на шаге, где k+1 позиций переходят границу блока, A увеличивается на единицу,
        # и чтение уезжает ровно на блок мимо записи. Прибор (FA2SM70_GDN_GRAN_DIAG) печатает
        # это прямо: `ctx=8190 seq=8194 m=4 start_тек=2 start_зап=1`.
        # Лечить это черновиком нельзя: при асинхронном планировании спекулятивный батч
        # ПАДДИРУЕТСЯ до k+1 (движок сам требует padded drafter batch), и снятие черновика
        # число позиций шага не меняет -- проверено рычагом FA2SM70_GRAN_ALL_OFF=1: шаги шли
        # с q=4 при пустом черновике.
        # Поэтому правится АДРЕС: в ту ячейку, откуда ядро возьмёт начальное состояние
        # (колонка num_accepted-1), кладётся блок, где состояние ЛЕЖИТ НА САМОМ ДЕЛЕ --
        # A_зап + num_accepted - 1, где A_зап=(ctx-num_accepted+k)//B -- опора ПРОШЛОГО шага.
        # Подстановка безусловна: когда опора не двигалась, A_зап==A_тек и в ячейку ложится
        # ровно то же значение, что там и было. Ветвлений нет -- значит переживает CUDA-граф.
        # [ДЕРЕВО, 05.09-2] Сдвиг чтения на W колонок нужен ТОЛЬКО SSM (состояния строк
        # ветви B лежат в колонках W+1..2W), и только при m>1: при m=1 принят один бонус,
        # его состояние -- у якоря, колонка 0. Свёртке сдвиг ПРОТИВОПОКАЗАН: её окно после
        # подмены из чернового слота живёт обычной цепной арифметикой (интеграционный тест
        # test_derevo_conv.py: ветви A и B, m=1..3 -- relL2 ~6e-08). Единый сдвиг, который
        # стоял здесь раньше, давал свёртке ряд [a1, b0, b1] -- подпись «слово!» из §128b.
        num_accepted_ssm = None
        num_accepted_conv = None
        if num_accepted_tokens is not None and num_spec_decodes > 0:
            try:
                from vllm.model_executor.models.qwen3_next import ДЕРЕВО_OFF as _ДО
                _б = _ДО.get("буф")
                # [СНЯТА СИНХРОНИЗАЦИЯ РАДИ ПЕЧАТИ, 08.09]
                # Здесь стояло условие `int(_б[0]) != 0` БЕЗ РЫЧАГА. `_б` -- тензор КАРТЫ,
                # и `int(...)` по нему -- полная синхронизация: хозяин ждёт карту на КАЖДОМ
                # построении метаданных, то есть каждый шаг, ради отладочной строки. Две
                # соседние диагностики ([ПРИЁМНИК], [ПОДСТАНОВКА]) уже стояли за `_ГРАН_ДИАГ`;
                # эта осталась открытой. Теперь она за тем же рычагом -- в бою ветка мертва
                # и ни одного обращения к карте не делает.
                if _ГРАН_ДИАГ:
                    _СЧЁТ["ssm_зов"] = _СЧЁТ.get("ssm_зов", 0) + 1
                    if _СЧЁТ["ssm_зов"] <= 6 or (_б is not None and int(_б[0]) != 0):
                        print(f"[BUILD ssm] зов={_СЧЁТ['ssm_зов']} буф={'есть' if _б is not None else 'НЕТ'} "
                              f"nacc0={int(num_accepted_tokens[0])} "
                              f"off0={int(_б[0]) if _б is not None else '-'}",
                              file=_sys.stderr, flush=True)
                if _б is not None:
                    _nб = min(int(num_accepted_tokens.shape[0]), int(_б.shape[0]))
                    # [ТОЧНОЕ ЛЕЧЕНИЕ ВИСЯЧЕГО УКАЗАТЕЛЯ, 08.09]
                    # Бисекция назвала ИМЕННО ЭТОТ буфер: удержание одного его
                    # даёт 3 соака без падения, а удержание таблицы блоков -- падение
                    # на первом. Механизм: буфер создавался лениво (мог попасть в
                    # ЗАХВАТ) и ЗАМЕНЯЛСЯ при нехватке -- старый отпускался, а его
                    # адрес запечён в графе, и повтор писал в чужую память.
                    # Лечение без вечного удержания: выделяем СРАЗУ под потолок
                    # (число запросов планировщика), а редкий больший случай
                    # обслуживаем ВРЕМЕННЫМ тензором, которого граф никогда не видит.
                    _потолок = max(int(self.decode_cudagraph_max_bs),
                                   int(getattr(self, "_потолок_запросов", 0) or 0), 1)
                    _буф_ssm = getattr(self, "_nacc_ssm_буф", None)
                    if _буф_ssm is None:
                        _буф_ssm = self._nacc_ssm_буф = torch.empty(
                            max(_nб, _потолок),
                            dtype=num_accepted_tokens.dtype,
                            device=num_accepted_tokens.device)
                    elif _буф_ssm.shape[0] < _nб:
                        # НЕ заменяем постоянный буфер: он, возможно, уже в графе.
                        _буф_ssm = torch.empty(
                            _nб, dtype=num_accepted_tokens.dtype,
                            device=num_accepted_tokens.device)
                    _буф_ssm[:_nб].copy_(num_accepted_tokens[:_nб])
                    _буф_ssm[:_nб] += _б[:_nб] * (num_accepted_tokens[:_nб] > 1)
                    num_accepted_ssm = _буф_ssm[:_nб]
                    # [ПРОТОКОЛ-РЕПЛЕЙ, 05.09] Сцепление окон показало: conv в графе видит
                    # nacc ПРОШЛОГО шага (подтверждено на трёх переходах). Граф захватил
                    # ПОСТОЯННЫЙ буфер self.num_accepted_tokens (кладёт переупаковщик), а
                    # build отдавал ЛОКАЛЬНЫЙ тензор -- буфер жил с лагом на шаг. Кладём
                    # свежее значение прямо здесь: адрес, который читает граф, обновлён.
                    _бк = getattr(self, "_nacc_conv_буф", None)
                    if _бк is not None:
                        _нб3 = min(_nб, int(_бк.shape[0]))
                        _бк[:_нб3].copy_(num_accepted_tokens[:_нб3])
                        num_accepted_conv = _бк[:_нб3]
            except Exception:
                num_accepted_ssm = None
        # ЦЕНА ПОДСТАНОВКИ ПЛАТИТСЯ ТОЛЬКО У ГРАНИЦЫ. Строитель метаданных зовётся по разу на
        # KV-группу (их десять), поэтому безусловная подстановка стоила 31.8 -> 25.9 ток/с
        # (-18 %). Сама она нужна на ~8 шагах из 4096, и близость границы видна ПО ПРОЦЕССОРНЫМ
        # длинам -- без единого обращения к карте. Метаданные строятся ВНЕ графа, поэтому
        # условность здесь законна: граф исполняет ядра и читает буфер уже исправленным.
        _спец_conv_блоки = _спец_conv_чт = _спец_conv_зап = None
        _гран_рядом_cpu = _ГРАН_ВСЕГДА
        if (_ГРАН_КОНВ_ВСЕГДА and mamba_all and spec_state_indices_tensor is not None
                and num_accepted_tokens is not None and num_spec_decodes > 0):
            try:
                # [ПЕРЕПИСАНО 06.09 ПОСЛЕ РАЗБОРА ПАДЕНИЯ]
                # Спекулятивная таблица -- это ВЫРОВНЕННАЯ таблица, обрезанная до
                # num_spec+1 колонок (`_индексы_состояний`: `_src[маска, :num_spec+1]`,
                # где `_src` -- align-таблица). Её КОЛОНКИ ОТНОСИТЕЛЬНЫЕ: колонка 0 --
                # это блок, с которого таблица выровнена, а не блок номер ноль.
                # Первая редакция подставляла в `block_idx_last_scheduled_token`
                # АБСОЛЮТНЫЙ номер блока (`(seq_len-1)//B`, у нас это бывало 61) и тем
                # самым читала строку таблицы далеко за её пятью колонками -- оттуда
                # приходил мусорный слот, а ядро координату ЗАПИСИ не проверяет
                # (маска строки 932 смотрит только токены и признаки). Наружу это
                # выходило как Xid 13 и смерть воркера. Отсюда правило: колонки для
                # спекулятивной свёртки считаются ОТНОСИТЕЛЬНО начала выравнивания.
                #
                # Выравнивание берётся по seq_lens (см. `aligned_block_table`), то есть
                # колонка 0 -- блок ПОСЛЕДНЕГО ЗАПЛАНИРОВАННОГО токена. Значит:
                #   писать  -> колонка 0;
                #   читать  -> колонка (блок последнего посчитанного) - (блок последнего
                #              запланированного) <= 0, а отрицательных колонок нет.
                # Поэтому пара выражается ТОЛЬКО при выравнивании ПО КОНТЕКСТУ
                # (FA2SM70_GDN_SPEC_CTX=1): там колонка 0 -- блок последнего посчитанного,
                # состояние лежит ровно в ней, а запись идёт в колонку сдвига 0 или 1.
                # Без этого рычага пару выражать нечем, и мы её НЕ СТРОИМ -- прежнее
                # поведение (один слот, колонка 0) остаётся в силе.
                if True:
                    # СВОЯ ТАБЛИЦА У СВЁРТКИ, А НЕ ОБЩИЙ РЫЧАГ. `FA2SM70_GDN_SPEC_CTX`
                    # менял ОБЕ таблицы сразу -- и свёртки, и рекуррента, -- и в одиночку
                    # ронял воркер (+482 Xid за 168 запросов). Здесь контекстное
                    # выравнивание строится ЛОКАЛЬНО и отдаётся ТОЛЬКО свёртке:
                    # колонка 0 -- блок последнего ПОСЧИТАННОГО токена (там и лежит
                    # состояние), колонки 1..num_spec -- блоки, куда шагнут k+1 токенов.
                    # Тогда чтение выражается нулём, а запись -- сдвигом 0..num_spec,
                    # то есть обе координаты заведомо внутри строки.
                    _Bк = self.kv_cache_spec.block_size
                    if spec_sequence_masks is not None and int(num_spec_decodes) != int(m.num_reqs):
                        _мк = spec_sequence_masks[: context_lens_tensor.shape[0]]
                        _ctxк = context_lens_tensor[_мк]
                        _seqк = m.seq_lens[_мк]
                    else:
                        _ctxк, _seqк = context_lens_tensor, m.seq_lens
                    _nк = min(int(spec_state_indices_tensor.shape[0]), int(_ctxк.shape[0]),
                              int(self._conv_чт_буф.shape[0]))
                    if _nк > 0:
                        _шир_сп = int(spec_state_indices_tensor.shape[1])
                        # ОДНА лесенка вместо двух: те же пять операций на СКЛЕЙКЕ двух
                        # рядов вместо пяти на каждый ряд (минус пять запусков ядер).
                        _об = torch.stack((_ctxк[:_nк], _seqк[:_nк])).to(torch.int64)
                        _об = ((_об - 1) // _Bк).clamp(min=0)
                        _ст_ctx, _ст_seq = _об[0], _об[1]
                        _зап_отн = (_ст_seq - _ст_ctx).clamp(min=0, max=_шир_сп - 1)
                        # Копии сюда НЕ пишем: ниже обе строки переписываются ещё раз
                        # (после зажима по ширине таблицы и сверки с выравниванием), то есть
                        # эта пара была мёртвой работой. Буфер чтения нулевой по построению.
                        # ОТКАТ 06.09: редакция «таблица на блок назад + поиск по номеру
                        # блока» давала цену перехода -0.01 нат, приёмку 3.13 и 80 ток/с --
                        # и РАЗРУШАЛА ответ: каждая генерация вырождалась в «!!!» после
                        # первого токена. Высокая приёмка была следствием вырождения, а не
                        # заслугой: черновику легко угадывать повтор одного токена. Закон
                        # «скорость без гейта качества -- ложь» сработал ровно здесь.
                        # Возвращена редакция §150: таблица выровнена ПО КОНТЕКСТУ,
                        # чтение -- колонка 0, запись -- сдвиг 0/1 со сверкой у движка.
                        _таб_к = mamba_get_block_table_tensor(
                            m.block_table_tensor, context_lens_tensor,
                            self.kv_cache_spec, "align")
                        if (spec_sequence_masks is not None
                                and int(num_spec_decodes) != int(m.num_reqs)):
                            _таб_к = _таб_к[spec_sequence_masks[: _таб_к.shape[0]]]
                        _вш_к = min(_шир_сп, int(_таб_к.shape[1]),
                                    int(self._conv_блоки_буф.shape[1]))
                        if _вш_к <= 0 or int(_таб_к.shape[0]) < _nк:
                            raise ValueError("узкая таблица свёртки")
                        self._conv_блоки_буф[:_nк, :_вш_к].copy_(_таб_к[:_nк, :_вш_к])
                        _падк = (spec_state_indices_tensor[:_nк, 0] == PAD_SLOT_ID)
                        self._conv_блоки_буф[:_nк][_падк] = PAD_SLOT_ID
                        if _nк < int(self._conv_блоки_буф.shape[0]):
                            self._conv_блоки_буф[_nк:] = PAD_SLOT_ID
                            self._conv_зап_буф[_nк:] = 0    # буфер чтения нулевой всегда
                        _зап_отн = _зап_отн.clamp(min=0, max=_вш_к - 1)
                        if aligned_block_table is not None:
                            _свер = aligned_block_table
                            if (spec_sequence_masks is not None
                                    and int(num_spec_decodes) != int(m.num_reqs)):
                                _свер = _свер[spec_sequence_masks[: _свер.shape[0]]]
                            if int(_свер.shape[0]) >= _nк and int(_свер.shape[1]) > 0:
                                _цель = _таб_к[:_nк].gather(1, _зап_отн.unsqueeze(1)).squeeze(1)
                                _ок = (_цель == _свер[:_nк, 0])
                                _зап_отн = torch.where(_ок, _зап_отн,
                                                       torch.zeros_like(_зап_отн))
                        self._conv_зап_буф[:_nк].copy_(_зап_отн.to(torch.int32))
                        _спец_conv_блоки = self._conv_блоки_буф[:_nк, :_вш_к]
                        _спец_conv_чт = self._conv_чт_буф[:_nк]
                        _спец_conv_зап = self._conv_зап_буф[:_nк]
            except Exception:
                _спец_conv_блоки = _спец_conv_чт = _спец_conv_зап = None
        if (_ГРАН_ИСТОК and mamba_all and spec_state_indices_tensor is not None
                and num_spec_decodes > 0):
            try:
                _Bc = self.kv_cache_spec.block_size
                _slc = m.seq_lens_cpu[: int(m.num_reqs)]
                _qlc = (query_start_loc_cpu[1 : int(m.num_reqs) + 1]
                        - query_start_loc_cpu[: int(m.num_reqs)])
                _ctxc = (_slc - _qlc).to(torch.int64)
                # ШИРИНА ОКНА -- ЗАМЕРОМ, А НЕ ИЗ ФОРМУЛЫ. Расширение до 16*(k+1)=64
                # ОТВЕРГНУТО: базовый промпт остался 12/12, а сдвинутый упал с 5/12 до
                # 1/12. Значит лишнее срабатывание НЕ безвредно (подстановка вне
                # границы подсовывает блок из истории, а он совпадает с текущим не
                # всегда), и запас окна -- не свободный параметр. Оставлено 2*(k+1) с
                # ДВУСТОРОННЕЙ проверкой (см. ниже): это строго лучше прежнего в обоих
                # случаях (12/12 против 7/12 и 5/12 против 3/12).
                _окно = 2 * (self.num_spec + 1)
                # ОКНО ДВУСТОРОННЕЕ. Первая редакция смотрела ТОЛЬКО ВПЕРЁД
                # (ctx-1 против ctx+2(k+1)) и переставала срабатывать сразу ПОСЛЕ
                # перехода -- а подстановка нужна ещё несколько шагов: состояние
                # принятой позиции продолжает лежать в ПРЕЖНЕМ блоке. Замер
                # (гейт границы, 12 прогонов под нагрузкой, temp0):
                #   окно вперёд   -- 7/12 полных, обрывы на +4..+13 токенов ЗА границей;
                #   подстановка КАЖДЫЙ шаг (FA2SM70_GRAN_ALWAYS=1) -- 12/12 и детерминизм;
                #   без спекуляции -- 12/12 (спекуляция -- необходимое условие).
                # То есть лечение было верным, но не применялось на том шаге, ради
                # которого написано. Симметричное окно стоит столько же: срабатывает
                # на ~16 шагах из 4096 вместо ~8.
                _гран_рядом_cpu = _ГРАН_ВСЕГДА or bool((((_ctxc - _окно - 1) // _Bc)
                                        != ((_ctxc + _окно) // _Bc)).any())
            except Exception:
                _гран_рядом_cpu = True   # не смогли определить -- работаем как раньше
        if (_ГРАН_ИСТОК and _гран_рядом_cpu and mamba_all
                and spec_state_indices_tensor is not None
                and num_accepted_tokens is not None and num_spec_decodes > 0):
            _Bг = self.kv_cache_spec.block_size
            if spec_sequence_masks is not None:
                _мс = spec_sequence_masks[: block_table_tensor.shape[0]]
                _полн = block_table_tensor[_мс]
                _ctxг = context_lens_tensor[_мс]
                _seqг = m.seq_lens[_мс]
            else:
                _полн = block_table_tensor
                _ctxг = context_lens_tensor
                _seqг = m.seq_lens
            _nг = min(spec_state_indices_tensor.shape[0], _полн.shape[0],
                      _ctxг.shape[0], int(num_accepted_tokens.shape[0]))
            if _nг > 0 and _полн.shape[1] > 0:
                _mг = num_accepted_tokens[:_nг].to(torch.int64).clamp(min=1)
                _A_тек = (((_seqг[:_nг].to(torch.int64) - 1) // _Bг)).clamp(min=0)
                # [ОДНО ПРИВЕДЕНИЕ ВМЕСТО ЧЕТЫРЁХ, 08.09] `_ctxг[:_nг].to(int64)` стояло в
                # этом блоке ЧЕТЫРЕ раза (в _A_зап, _A_без, _ctx64г и в снимке истории).
                # Значение одно и то же -- считаем один раз.
                _ctx64 = _ctxг[:_nг].to(torch.int64)
                # `_A_зап` НУЖЕН ТОЛЬКО ДИАГНОСТИКЕ (единственный потребитель -- печать
                # [ПОДСТАНОВКА] под `_ГРАН_ДИАГ`). Считался безусловно: четыре запуска ядер
                # на каждом шаге ради строки, которой в бою нет.
                _A_зап = ((( _ctx64 - _mг + self.num_spec) // _Bг).clamp(min=0)
                          if _ГРАН_ДИАГ else None)
                # ИСТОЧНИК БЕРЁТСЯ ИЗ ИСТОРИИ, А НЕ ИЗ ДОГАДКИ О ПРОШЛОМ ШАГЕ.
                # Формула `A_зап=(ctx-m+k)//B` верна лишь если прошлый шаг был спекулятивным
                # на полную глубину. После префилла это не так: состояние там записано
                # неспекулятивным путём в блок (ctx-1)//B, и догадка давала блок 0 -- гейт
                # «17*23» отвечал мусором. Поэтому опора прошлого шага ЗАПОМИНАЕТСЯ, а её
                # пригодность сверяется по ctx: он обязан лежать в (ctx_пред, ctx_пред+q_пред].
                # Не сошлось -- берём (ctx-1)//B, то есть блок последнего посчитанного.
                _A_без = ((_ctx64 - 1) // _Bг).clamp(min=0)
                # [ДЕРЕВО, 05.09] СТОЛБЕЦ СОСТОЯНИЯ СДВИНУТ ВЕТВЬЮ. Состояние позиции j
                # рекуррент пишет в столбец СТРОКИ, а при ветви B позиция j -- это строка
                # j+W шага. Подстановка у границы брала столбец m-1 (раскладка ветви A) и
                # подсовывала ветви B блок чужой строки; вне границы это не видно, потому
                # порча шла редкими вспышками посреди ответа и запекалась в префикс-кэш.
                # Сдвиг берётся из того же off-буфера, что и num_accepted_ssm.
                _mк = _mг
                if (num_accepted_ssm is not None
                        and int(num_accepted_ssm.shape[0]) >= _nг):
                    _mк = num_accepted_ssm[:_nг].to(torch.int64).clamp(min=1)
                _ист = _A_без + _mк - 1
                # ИСТОРИЯ ХРАНИТ САМУ ТАБЛИЦУ ПРОШЛОГО ШАГА, А НЕ ОПОРУ. Опора описывает
                # адрес записи только пока таблицу никто не правил; но её правит подстановка
                # приёмника (ниже), и тогда формула по опоре указывает мимо. Таблица же
                # хранит ФАКТИЧЕСКИЕ адреса: состояние позиции j лежит там, где стояла
                # колонка j прошлого шага. Пригодность истории сверяется по ctx.
                _ист = _ист.clamp(min=0, max=_полн.shape[1] - 1)
                _знач = _полн[:_nг].gather(1, _ист.unsqueeze(1))
                _кол = (_mк - 1).clamp(min=0,
                                       max=spec_state_indices_tensor.shape[1] - 1)
                _таб_ист = getattr(self, "_гран_таб_пред", None)
                _ctx_ист = getattr(self, "_гран_ctx_пред", None)
                _q_ист = int(getattr(self, "_гран_q_пред", 0) or 0)
                if (_таб_ист is not None and _ctx_ист is not None
                        and _таб_ист.shape[0] == _nг and _ctx_ист.shape[0] == _nг
                        and _таб_ист.shape[1] == spec_state_indices_tensor.shape[1]
                        and _q_ист > 0):
                    _годно = ((_ctx64 > _ctx_ист)
                              & (_ctx64 <= _ctx_ист + _q_ист)).unsqueeze(1)
                    _знач = torch.where(
                        _годно, _таб_ист.gather(1, _кол.unsqueeze(1)), _знач
                    )
                # ОТКАТ 08.09: постоянный буфер здесь дал ХУЖЕ (фаза Б 1.18 -> 1.46 на трёх
                # повторах). Причина не разобрана, но замер однозначен, а правка была ради
                # скорости -- значит она отменяется. Клон возвращён.
                self._гран_ctx_пред = _ctx64.detach().clone()
                self._гран_q_пред = int(self.num_spec) + 1
                # БЕЗ КЛОНА. Клон разрывает связь с ПОСТОЯННЫМ буфером метаданных: полный
                # граф читает адреса своих буферов, и снимок с новым адресом он не видит --
                # генерация вырождалась в мусор (гейт «17*23» отвечал иероглифами). Пишем
                # на месте: буфер и так перезаписывается каждым шагом.
                spec_state_indices_tensor[:_nг].scatter_(
                    1, _кол.unsqueeze(1), _знач.to(spec_state_indices_tensor.dtype)
                )
                # ---- КОЛОНКА 0: СОСТОЯНИЕ СВЁРТКИ ---------------------------------------
                # Спекулятивная ветка свёртки берёт `spec_state_indices_tensor[:, 0]` -- один
                # слот, без пары «читать/писать», которая есть у обычного декода. Колонка 0 --
                # это блок опоры, и она уезжает на границе ровно так же, как колонка чтения
                # SSM. Обрывы лечила правка SSM, а свёртка продолжала терять окно на каждой
                # границе: режим 'all' со спекуляцией давал стабильные 106 токенов и 7 складов
                # из 8, тогда как 'none' и 'all' БЕЗ спекуляции -- 124 токена и 8 из 8.
                # Кладём в колонку 0 тот блок, где свёрточное состояние лежит на самом деле --
                # то есть колонку 0 таблицы прошлого шага. Запись пойдёт туда же, и следующий
                # шаг снова возьмёт её из истории: связка самосогласована.
                # ЗАМЕР ОТВЕРГ подстановку колонки 0: без нагрузки ответы перестали быть
                # одинаковыми (174/188/124/400/400/106/113/108) и полных стало 1 из 8 против
                # 1 из 10 при стабильных 106. Свёртке нужна не подмена одной ячейки, а ПАРА
                # указателей (ниже) -- её ядро это умеет. Рычаг оставлен выключенным.
                if (_ГРАН_КОНВ0 and _таб_ист is not None and _ctx_ист is not None
                        and _таб_ист.shape[0] == _nг
                        and _таб_ист.shape[1] == spec_state_indices_tensor.shape[1]
                        and _q_ист > 0):
                    _ноль = torch.zeros_like(_кол).unsqueeze(1)
                    _тек0 = spec_state_indices_tensor[:_nг].gather(1, _ноль)
                    _ист0 = _таб_ист.gather(1, _ноль).to(_тек0.dtype)
                    spec_state_indices_tensor[:_nг].scatter_(
                        1, _ноль, torch.where(_годно, _ист0, _тек0)
                    )
                # ---- ПРИЁМНИК: состояние на КОНЕЦ блока -- в сам блок --------------------
                # Спекулятивное ядро пишет состояние позиции j в колонку j, а колонка 0 --
                # это блок последнего ЗАПЛАНИРОВАННОГО токена. Значит в настоящий блок
                # ложится состояние ПЕРВОЙ позиции шага, и блок, который на этом шаге
                # закрывается, сохраняет в кэш недосчитанное состояние: при попадании в
                # префикс-кэш ответ уезжает (замер: без кэша 12/12 полных и ответ побайтово
                # один и тот же, с кэшем 10-16 из 20). Кладём в колонку той позиции, что
                # закрывает блок, номер САМОГО блока -- тогда ядро запишет туда состояние
                # ровно на конец блока. Колонку чтения не трогаем: она главнее.
                if _ГРАН_ПРИЁМ:
                    _ctx64п = _ctxг[:_nг].to(torch.int64)
                    _бл = (_ctx64п // _Bг)
                    _jзв = (_бл + 1) * _Bг - 1 - _ctx64п
                    _шир = spec_state_indices_tensor.shape[1]
                    _годно_п = ((_jзв >= 0) & (_jзв < _шир)
                                & (_jзв != (_mг - 1))).unsqueeze(1)
                    _колп = _jзв.clamp(min=0, max=_шир - 1).unsqueeze(1)
                    _значп = _полн[:_nг].gather(
                        1, _бл.clamp(min=0, max=_полн.shape[1] - 1).unsqueeze(1)
                    )
                    _текущ = spec_state_indices_tensor[:_nг].gather(1, _колп)
                    spec_state_indices_tensor[:_nг].scatter_(
                        1, _колп,
                        torch.where(_годно_п, _значп.to(_текущ.dtype), _текущ),
                    )
                    if _ГРАН_ДИАГ:
                        _срп = int(_годно_п.sum())
                        if _срп:
                            _СЧЁТ["приём"] = _СЧЁТ.get("приём", 0) + _срп
                            print(f"[ПРИЁМНИК] строк={_срп} j*={int(_jзв[0])} "
                                  f"блок={int(_бл[0])} всего={_СЧЁТ['приём']}",
                                  file=_sys.stderr, flush=True)
                if _ГРАН_КОНВ:
                    # Опора ПРОШЛОГО шага как колонка полной таблицы. При негодной истории
                    # берём текущую -- тогда поведение в точности прежнее.
                    _A_пред_к = getattr(self, "_гран_Aк_пред", None)
                    _чт = _A_тек
                    if (_A_пред_к is not None and _A_пред_к.shape[0] == _nг
                            and _q_ист > 0 and _ctx_ист is not None
                            and _ctx_ист.shape[0] == _nг):
                        _чт = torch.where(_годно.squeeze(1), _A_пред_к, _A_тек)
                    _вш = min(_полн.shape[1], self._conv_блоки_буф.shape[1])
                    _вб_н = min(_nг, self._conv_чт_буф.shape[0])
                    self._conv_чт_буф[:_вб_н].copy_(_чт[:_вб_н].to(torch.int32))
                    self._conv_зап_буф[:_вб_н].copy_(_A_тек[:_вб_н].to(torch.int32))
                    _спец_conv_чт = self._conv_чт_буф[:_вб_н]
                    _спец_conv_зап = self._conv_зап_буф[:_вб_н]
                    # ПАДДИНГОВЫЕ СТРОКИ ОБЯЗАНЫ ОСТАТЬСЯ ПОМЕЧЕННЫМИ. Прежний вызов брал
                    # колонку spec-таблицы, где фиктивные строки несут PAD_SLOT_ID, и ядро их
                    # пропускало (`USE_PAD_SLOT`). В полной таблице блоков на их месте нули,
                    # и ядро принималось считать несуществующий запрос -- выход за буфер
                    # всплывал как illegal memory access в чужом ядре через сотни строк.
                    self._conv_блоки_буф[:_вб_н, :_вш].copy_(_полн[:_вб_н, :_вш])
                    _пад = (spec_state_indices_tensor[:_вб_н, 0] == PAD_SLOT_ID)
                    self._conv_блоки_буф[:_вб_н][_пад] = PAD_SLOT_ID
                    _спец_conv_блоки = self._conv_блоки_буф[:_вб_н, :_вш]
                    self._гран_Aк_пред = _A_тек.detach().clone()
                # Историю снимаем ПОСЛЕ всех подстановок -- она обязана хранить фактические
                # адреса записи, иначе следующий шаг прочитает не оттуда.
                # [ПОСТОЯННЫЙ БУФЕР ВМЕСТО КЛОНА, 08.09] `clone().to(int64)` выделял НОВЫЙ
                # тензор на каждом шаге -- это и работа распределителя, и лишняя копия в
                # горячем пути. Буфер выделяется один раз под потолок и НЕ заменяется
                # (закон о висячем указателе: заменять то, что могло попасть в граф, нельзя;
                # здесь тензор в графы не попадает, но правило дешевле соблюсти).
                _ист_буф = getattr(self, "_гран_ист_буф", None)
                if (_ист_буф is None or _ист_буф.shape[0] < _nг
                        or _ист_буф.shape[1] != spec_state_indices_tensor.shape[1]):
                    _потг = max(int(self.decode_cudagraph_max_bs),
                                int(getattr(self, "_потолок_запросов", 0) or 0), _nг, 1)
                    _ист_буф = self._гран_ист_буф = torch.empty(
                        (_потг, spec_state_indices_tensor.shape[1]),
                        dtype=torch.int64, device=spec_state_indices_tensor.device)
                _ист_буф[:_nг].copy_(spec_state_indices_tensor[:_nг])
                self._гран_таб_пред = _ист_буф[:_nг]
                if _ГРАН_ДИАГ:
                    _пр = int((_A_тек != _A_зап).sum())
                    if _пр:
                        _СЧЁТ["подстановок"] = _СЧЁТ.get("подстановок", 0) + _пр
                        print(f"[ПОДСТАНОВКА] строк={_пр} A_тек={int(_A_тек[0])} "
                              f"A_зап={int(_A_зап[0])} всего={_СЧЁТ['подстановок']}",
                              file=_sys.stderr, flush=True)

        attn_metadata = GDNAttentionMetadata(
            seq_lens_для_обновления=m.seq_lens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_spec_decodes=num_spec_decodes,
            num_spec_decode_tokens=num_spec_decode_tokens,
            num_actual_tokens=m.num_actual_tokens,
            has_initial_state=has_initial_state,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            spec_query_start_loc=spec_query_start_loc,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_state_indices_tensor=spec_state_indices_tensor,
            non_spec_state_indices_tensor=non_spec_state_indices_tensor,
            spec_sequence_masks=spec_sequence_masks,
            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
            spec_state_slot_selectors=spec_state_slot_selectors,
            ddtree_parent_ids=ddtree_parent_ids,
            ddtree_num_tree_tokens_cpu=ddtree_num_tree_tokens_cpu,
            num_accepted_ssm=num_accepted_ssm,
            num_accepted_conv=num_accepted_conv,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
            block_idx_last_computed_token=block_idx_last_computed_token,
            block_idx_last_scheduled_token=block_idx_last_scheduled_token,
            block_idx_first_scheduled_token=block_idx_first_scheduled_token,
            num_computed_tokens_ns=num_computed_tokens_ns,
            num_computed_tokens_ns_cpu=num_computed_tokens_ns_cpu,
            non_spec_query_start_loc_cpu=(
                non_spec_query_start_loc_cpu if mamba_all and num_prefills > 0 else None
            ),
            mamba_block_size=(self.kv_cache_spec.block_size if mamba_all else 0),
            spec_conv_блоки=_спец_conv_блоки,
            spec_conv_чт=_спец_conv_чт,
            spec_conv_зап=_спец_conv_зап,
        )
        _гф("Г сборка объекта", _тГ)
        if ddtree_parent_ids is not None and _ddtree_trace_path():
            _write_ddtree_trace_event(
                "gdn_metadata",
                {
                    "for_cudagraph_capture": for_cudagraph_capture,
                    "num_prefills": num_prefills,
                    "num_decodes": num_decodes,
                    "num_spec_decodes": num_spec_decodes,
                    "num_spec_decode_tokens": num_spec_decode_tokens,
                    "num_actual_tokens": m.num_actual_tokens,
                    "seq_lens": _trace_tensor(m.seq_lens),
                    "query_start_loc_cpu": _trace_tensor(query_start_loc_cpu),
                    "num_accepted_tokens": _trace_tensor(num_accepted_tokens),
                    "spec_state_slot_selectors": _trace_tensor(
                        spec_state_slot_selectors
                    ),
                    "spec_query_start_loc": _trace_tensor(spec_query_start_loc),
                    "spec_state_indices_tensor": _trace_tensor(
                        spec_state_indices_tensor
                    ),
                    "current_state_block_ids": _trace_tensor(current_state_block_ids),
                    "ddtree_parent_ids": _trace_tensor(ddtree_parent_ids),
                    "ddtree_num_tree_tokens_cpu": _trace_tensor(
                        ddtree_num_tree_tokens_cpu
                    ),
                },
            )
        if self.use_spec_decode and envs.VLLM_SM70_QWEN_GDN_SPEC_CORE_OP:
            profile_register_t0 = time.perf_counter() if metadata_profile else 0.0
            register_gdn_spec_metadata_tensors(
                self.layer_names,
                gdn_spec_metadata_tensors(attn_metadata, query_start_loc.device),
            )
            if metadata_profile:
                profile_register_ms = (
                    time.perf_counter() - profile_register_t0
                ) * 1000.0
        if metadata_profile:
            logger.info(
                "DFLASH_DDTREE_METADATA_PROFILE gdn_build total_ms=%.3f "
                "state_contract_ms=%.3f graph_buffers_ms=%.3f "
                "register_ms=%.3f num_prefills=%d num_decodes=%d "
                "num_spec_decodes=%d num_spec_decode_tokens=%d "
                "num_actual_tokens=%d full_graph=%s ddtree=%s layers=%d",
                (time.perf_counter() - metadata_profile_t0) * 1000.0,
                profile_state_contract_ms,
                profile_graph_buffers_ms,
                profile_register_ms,
                num_prefills,
                num_decodes,
                num_spec_decodes,
                num_spec_decode_tokens,
                m.num_actual_tokens,
                self.use_full_cuda_graph,
                ddtree_parent_ids is not None,
                len(self.layer_names),
            )
        if os.getenv("VLLM_SM70_DUMP_GDN_STATE_TABLE_DIR"):

            def _cpu(t: torch.Tensor | None) -> torch.Tensor | None:
                return None if t is None else t.detach().cpu()

            dump_path = _dump_sm70_gdn_state_table(
                {
                    "num_spec": self.num_spec,
                    "num_spec_state_tokens": self.num_spec_state_tokens,
                    "layer_names": self.layer_names,
                    "use_full_cuda_graph": self.use_full_cuda_graph,
                    "decode_cudagraph_max_bs": self.decode_cudagraph_max_bs,
                    "num_prefills": num_prefills,
                    "num_prefill_tokens": num_prefill_tokens,
                    "num_decodes": num_decodes,
                    "num_decode_tokens": num_decode_tokens,
                    "num_spec_decodes": num_spec_decodes,
                    "num_spec_decode_tokens": num_spec_decode_tokens,
                    "num_actual_tokens": m.num_actual_tokens,
                    "query_start_loc": _cpu(query_start_loc),
                    "query_start_loc_cpu": query_start_loc_cpu.detach().cpu(),
                    "seq_lens": _cpu(m.seq_lens),
                    "block_table_tensor": _cpu(block_table_tensor),
                    "current_state_block_ids": _cpu(current_state_block_ids),
                    "num_decode_draft_tokens_cpu": _cpu(num_decode_draft_tokens_cpu),
                    "spec_sequence_masks_cpu": _cpu(spec_sequence_masks_cpu),
                    "spec_sequence_masks": _cpu(spec_sequence_masks),
                    "num_accepted_tokens": _cpu(num_accepted_tokens),
                    "spec_query_start_loc": _cpu(spec_query_start_loc),
                    "non_spec_query_start_loc": _cpu(non_spec_query_start_loc),
                    "spec_token_indx": _cpu(spec_token_indx),
                    "non_spec_token_indx": _cpu(non_spec_token_indx),
                    "spec_state_indices_tensor": _cpu(spec_state_indices_tensor),
                    "non_spec_state_indices_tensor": _cpu(
                        non_spec_state_indices_tensor
                    ),
                    "ddtree_parent_ids": _cpu(ddtree_parent_ids),
                    "ddtree_num_tree_tokens_cpu": _cpu(ddtree_num_tree_tokens_cpu),
                },
                m.seq_lens,
                num_prefills,
                num_decodes,
            )
            if dump_path:
                logger.warning(
                    "Saved SM70 GDN state table diagnostics to %s", dump_path
                )
        if envs.VLLM_DFLASH_DEBUG_STATE_TABLE and self.use_spec_decode:

            def _cpu(t: torch.Tensor | None) -> torch.Tensor | None:
                return None if t is None else t.detach().cpu()

            dump_path = _dump_dflash_state_table(
                {
                    "num_spec": self.num_spec,
                    "num_spec_state_tokens": self.num_spec_state_tokens,
                    "layer_names": self.layer_names,
                    "use_full_cuda_graph": self.use_full_cuda_graph,
                    "decode_cudagraph_max_bs": self.decode_cudagraph_max_bs,
                    "num_prefills": num_prefills,
                    "num_prefill_tokens": num_prefill_tokens,
                    "num_decodes": num_decodes,
                    "num_decode_tokens": num_decode_tokens,
                    "num_spec_decodes": num_spec_decodes,
                    "num_spec_decode_tokens": num_spec_decode_tokens,
                    "num_actual_tokens": m.num_actual_tokens,
                    "query_start_loc": _cpu(query_start_loc),
                    "query_start_loc_cpu": query_start_loc_cpu.detach().cpu(),
                    "seq_lens": _cpu(m.seq_lens),
                    "block_table_tensor": _cpu(block_table_tensor),
                    "num_decode_draft_tokens_cpu": _cpu(num_decode_draft_tokens_cpu),
                    "spec_sequence_masks_cpu": _cpu(spec_sequence_masks_cpu),
                    "spec_sequence_masks": _cpu(spec_sequence_masks),
                    "num_accepted_tokens": _cpu(num_accepted_tokens),
                    "spec_query_start_loc": _cpu(spec_query_start_loc),
                    "non_spec_query_start_loc": _cpu(non_spec_query_start_loc),
                    "spec_token_indx": _cpu(spec_token_indx),
                    "non_spec_token_indx": _cpu(non_spec_token_indx),
                    "spec_state_indices_tensor": _cpu(spec_state_indices_tensor),
                    "non_spec_state_indices_tensor": _cpu(
                        non_spec_state_indices_tensor
                    ),
                    "ddtree_parent_ids": _cpu(ddtree_parent_ids),
                    "ddtree_num_tree_tokens_cpu": _cpu(ddtree_num_tree_tokens_cpu),
                }
            )
            logger.warning("Saved DFlash/GDN state table diagnostics to %s", dump_path)
        _задержка_хозяина()
        _гпечать()
        return attn_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ):
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with Mamba.
        """
        m = common_attn_metadata

        assert (
            m.num_reqs <= self.decode_cudagraph_max_bs
            and m.num_actual_tokens <= self.decode_cudagraph_max_bs
        ), (
            f"GDN only supports decode-only full CUDAGraph capture. "
            f"Make sure batch size ({m.num_reqs}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs}), "
            f"and number of tokens ({m.num_actual_tokens}) <= "
            f"cudagraph capture sizes ({self.decode_cudagraph_max_bs})."
        )

        num_accepted_tokens = torch.diff(m.query_start_loc)
        num_decode_draft_tokens_cpu = (num_accepted_tokens - 1).cpu()
        spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0

        return self.build(
            common_prefix_len=0,
            common_attn_metadata=m,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            spec_sequence_masks_cpu=spec_sequence_masks_cpu,
            for_cudagraph_capture=True,
        )
