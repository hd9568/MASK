#!/usr/bin/env python3
"""External SpMM backend bridge for end2end GCN.

Design goals:
- Use the src_fp32 DTC implementation first:
    /workspace/sparsene/examples/src_fp32/dtc/testbed
- Keep a stable extension interface for variants:
    - DTC base / strict_lb / multi_binding
    - ACC / BITBSR / SR_BCRS
- Add FlashSparse bridge with the same external entry style
- Default to no CSR fallback in benchmark mode (can be re-enabled by env).
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import importlib
import os
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch


Tensor = torch.Tensor

_WARNED_MESSAGES = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED_MESSAGES:
        print(f"[WARN] {msg}")
        _WARNED_MESSAGES.add(msg)


def _csr_fallback_enabled() -> bool:
    # Default is disabled to ensure end2end profiling always reflects target kernels.
    return os.getenv("SPARSENE_ALLOW_CSR_FALLBACK", "0") == "1"


def _fallback_or_raise(
    msg: str,
    rowptr: Tensor,
    colind: Tensor,
    values: Tensor,
    dense: Tensor,
    num_rows: int,
) -> Tensor:
    if _csr_fallback_enabled():
        _warn_once(f"{msg}; fallback to torch SpMM")
        return _torch_spmm(rowptr, colind, values, dense, num_rows)
    raise RuntimeError(
        f"{msg}. CSR fallback is disabled. "
        "Set SPARSENE_ALLOW_CSR_FALLBACK=1 to re-enable fallback."
    )


def _parse_tile_env(tile_env: str) -> Optional[int]:
    raw = os.getenv(tile_env, "auto").strip().lower()
    if raw in {"", "auto"}:
        return None
    if raw in {"16", "32", "64"}:
        return int(raw)
    _warn_once(f"Invalid {tile_env}={raw}, expected auto/16/32/64; using auto")
    return None


def _select_tile_b(feat_dim: int, tile_env: str) -> int:
    forced = _parse_tile_env(tile_env)
    candidates = [64, 32, 16] if forced is None else [forced]

    best_tile = candidates[0]
    best_pad = ((feat_dim + best_tile - 1) // best_tile) * best_tile - feat_dim
    for tile in candidates[1:]:
        pad = ((feat_dim + tile - 1) // tile) * tile - feat_dim
        # Prefer less padding; if tied, prefer larger tile.
        if pad < best_pad or (pad == best_pad and tile > best_tile):
            best_tile = tile
            best_pad = pad
    return best_tile


def _pad_dense_to_tile(dense: Tensor, tile_b: int) -> Tuple[Tensor, int]:
    out_cols = int(dense.shape[1])
    padded_cols = ((out_cols + tile_b - 1) // tile_b) * tile_b
    if padded_cols == out_cols:
        return dense.contiguous(), out_cols
    dense_padded = torch.nn.functional.pad(dense, (0, padded_cols - out_cols)).contiguous()
    return dense_padded, out_cols


@contextmanager
def _nvtx_range(name: str):
    pushed = False
    if os.getenv("SPARSENE_ENABLE_NVTX", "1") == "1" and torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_push(name)
            pushed = True
        except Exception:
            pushed = False
    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass


def _torch_spmm(rowptr: Tensor, colind: Tensor, values: Tensor, dense: Tensor, num_rows: int) -> Tensor:
    with _nvtx_range("spmm.fallback.torch_csr"):
        csr = torch.sparse_csr_tensor(
            crow_indices=rowptr,
            col_indices=colind,
            values=values,
            size=(num_rows, num_rows),
            device=dense.device,
        )
        return torch.sparse.mm(csr, dense)


@dataclass(frozen=True)
class BackendConfig:
    name: str
    variant: str


class BackendBase:
    def run(
        self,
        rowptr: Tensor,
        colind: Tensor,
        values: Tensor,
        dense: Tensor,
        num_rows: int,
        feat_dim: int,
        num_nodes_ori: int,
    ) -> Tensor:
        raise NotImplementedError


class TorchRefBackend(BackendBase):
    def run(
        self,
        rowptr: Tensor,
        colind: Tensor,
        values: Tensor,
        dense: Tensor,
        num_rows: int,
        feat_dim: int,
        num_nodes_ori: int,
    ) -> Tensor:
        return _torch_spmm(rowptr, colind, values, dense, num_rows)


class DTCBackend(BackendBase):
    """DTC bridge using src_fp32 testbed extension module.

    Expected source root:
      /workspace/sparsene/examples/src_fp32/dtc/testbed
    """

    def __init__(self, variant: str) -> None:
        self.variant = variant.strip().lower() or "base"
        self._module = None
        self._cache: Dict[Tuple[int, int, int, int, str], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]] = {}
        self._i32_index_cache: Dict[Tuple[int, Tuple[int, ...], str, torch.dtype], Tensor] = {}
        self._preprocess_func: Optional[Callable[..., Any]] = None
        self._run_func_default: Optional[Callable[..., Any]] = None
        self._run_func_balance: Optional[Callable[..., Any]] = None
        self._run_func_default_name: Optional[str] = None
        self._run_func_balance_name: Optional[str] = None
        self._source_root = Path(
            os.getenv(
                "SPARSENE_DTC_SOURCE_ROOT",
                "/workspace/sparsene/examples/src_fp32/dtc/testbed",
            )
        ).resolve()
        self._module_name = os.getenv("SPARSENE_DTC_MODULE_NAME", "DTCSpMM").strip() or "DTCSpMM"
        self._load_module()

    @staticmethod
    def _try_preload_shared_libs() -> None:
        preload_items = []

        # User-provided absolute library paths, separated by ':'.
        user_preload = os.getenv("SPARSENE_DTC_PRELOAD_LIBS", "").strip()
        if user_preload:
            preload_items.extend([x for x in user_preload.split(":") if x])

        for lib_path in preload_items:
            path_obj = Path(lib_path)
            if not path_obj.exists():
                continue
            try:
                ctypes.CDLL(str(path_obj), mode=ctypes.RTLD_GLOBAL)
            except Exception as exc:
                _warn_once(f"Preload shared lib failed: {lib_path} ({exc})")

    @staticmethod
    def _iter_build_lib_dirs(source_root: Path) -> Iterable[Path]:
        # Typical setuptools output: build/lib.linux-x86_64-3.10
        build_dir = source_root / "build"
        if not build_dir.exists():
            return []
        out = []
        for child in build_dir.glob("lib*"):
            if child.is_dir():
                out.append(child)
        return out

    def _push_module_search_paths(self) -> None:
        candidates = []

        user_module_path = os.getenv("SPARSENE_DTC_MODULE_PATH", "").strip()
        if user_module_path:
            candidates.append(Path(user_module_path))

        candidates.append(self._source_root)
        candidates.extend(self._iter_build_lib_dirs(self._source_root))

        for candidate in candidates:
            if candidate.exists():
                sys.path.insert(0, str(candidate))

    def _module_is_from_allowed_root(self, module: Any) -> bool:
        module_file = getattr(module, "__file__", "") or ""
        if not module_file:
            return False

        allow_non_src = os.getenv("SPARSENE_DTC_ALLOW_NON_SRC_FP32", "0") == "1"
        if allow_non_src:
            return True

        try:
            module_path = Path(module_file).resolve()
        except Exception:
            return False

        try:
            module_path.relative_to(self._source_root)
            return True
        except Exception:
            return False

    @staticmethod
    def _find_first_attr(module: Any, names: Iterable[str]) -> Optional[str]:
        for name in names:
            if hasattr(module, name):
                return name
        return None

    def _to_i32_cached(self, t: Tensor) -> Tensor:
        if t.dtype == torch.int32 and t.is_contiguous():
            return t
        key = (int(t.data_ptr()), tuple(t.shape), str(t.device), t.dtype)
        cached = self._i32_index_cache.get(key)
        if cached is not None:
            return cached
        converted = t.to(torch.int32).contiguous()
        self._i32_index_cache[key] = converted
        return converted

    def _resolve_entrypoints(self) -> None:
        self._preprocess_func = None
        self._run_func_default = None
        self._run_func_balance = None
        self._run_func_default_name = None
        self._run_func_balance_name = None
        if self._module is None:
            return

        preprocess_candidates: Dict[str, Tuple[str, ...]] = {
            "base": ("preprocess_gpu",),
            "strict_lb": ("preprocess_gpu_strict_lb", "preprocess_gpu"),
            "multi_binding": ("preprocess_gpu_multi_binding", "preprocess_gpu"),
        }
        run_candidates: Dict[str, Tuple[str, ...]] = {
            "base": ("DTCSpMM_gcn", "run_DTCSpMM"),
            "strict_lb": ("run_DTCSpMM_strict_lb", "run_DTCSpMM"),
            "multi_binding": ("run_DTCSpMM_multi_binding", "run_DTCSpMM"),
        }
        run_balance_candidates: Dict[str, Tuple[str, ...]] = {
            "base": ("run_DTCSpMM_balance", "run_DTCSpMM"),
            "strict_lb": (
                "run_DTCSpMM_strict_lb_balance",
                "run_DTCSpMM_balance",
                "run_DTCSpMM_strict_lb",
                "run_DTCSpMM",
            ),
            "multi_binding": (
                "run_DTCSpMM_multi_binding_balance",
                "run_DTCSpMM_balance",
                "run_DTCSpMM_multi_binding",
                "run_DTCSpMM",
            ),
        }

        variant = self.variant if self.variant in preprocess_candidates else "base"
        preprocess_name = self._find_first_attr(self._module, preprocess_candidates[variant])
        if preprocess_name is not None:
            self._preprocess_func = getattr(self._module, preprocess_name)
        else:
            _warn_once(
                f"No preprocess function for DTC variant={self.variant} in module={self._module_name}"
            )

        run_name = self._find_first_attr(self._module, run_candidates[variant])
        if run_name is not None:
            self._run_func_default = getattr(self._module, run_name)
            self._run_func_default_name = run_name

        run_balance_name = self._find_first_attr(self._module, run_balance_candidates[variant])
        if run_balance_name is not None:
            self._run_func_balance = getattr(self._module, run_balance_name)
            self._run_func_balance_name = run_balance_name

    def _load_module(self) -> None:
        self._try_preload_shared_libs()
        self._push_module_search_paths()

        # Only accept modules from src_fp32 root by default.
        try:
            module = importlib.import_module(self._module_name)
            if not self._module_is_from_allowed_root(module):
                module_file = getattr(module, "__file__", "unknown")
                _warn_once(
                    "Ignoring non-src_fp32 DTC module: "
                    f"{module_file}. Set SPARSENE_DTC_ALLOW_NON_SRC_FP32=1 to override."
                )
                self._module = None
                return
            self._module = module
            self._resolve_entrypoints()
            return
        except Exception as exc:
            _warn_once(
                "src_fp32 DTC module import failed "
                f"(module={self._module_name}, root={self._source_root}): {exc}"
            )

        _warn_once("src_fp32 DTC extension unavailable, fallback to torch SpMM")

    def _get_preprocessed(
        self,
        rowptr: Tensor,
        colind: Tensor,
        num_rows: int,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]]:
        if self._module is None:
            return None
        if not rowptr.is_cuda or not colind.is_cuda:
            return None

        block_h = int(os.getenv("SPARSENE_DTC_BLK_H", "16"))
        block_w = int(os.getenv("SPARSENE_DTC_BLK_W", "8"))

        # data_ptr is enough for per-run caching in this end2end script.
        cache_key = (
            int(rowptr.data_ptr()),
            int(colind.data_ptr()),
            int(num_rows),
            int(colind.numel()),
            f"{block_h}x{block_w}",
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        row_windows = (num_rows + block_h - 1) // block_h
        edge_num = int(colind.numel())

        block_partition = torch.zeros(row_windows, dtype=torch.int32, device=rowptr.device)
        edge_to_column = torch.zeros(edge_num, dtype=torch.int32, device=rowptr.device)
        edge_to_row = torch.zeros(edge_num, dtype=torch.int32, device=rowptr.device)

        if self._preprocess_func is None:
            return None

        prep = self._preprocess_func(
            self._to_i32_cached(colind),
            self._to_i32_cached(rowptr),
            int(num_rows),
            int(block_h),
            int(block_w),
            block_partition,
            edge_to_column,
            edge_to_row,
        )
        self._cache[cache_key] = prep
        return prep

    def _run_variant(self, rowptr: Tensor, colind: Tensor, values: Tensor, dense: Tensor, num_rows: int) -> Tensor:
        prep = self._get_preprocessed(rowptr, colind, num_rows)
        if prep is None:
            return _fallback_or_raise(
                f"DTC preprocess unavailable for variant={self.variant}",
                rowptr,
                colind,
                values,
                dense,
                num_rows,
            )

        row_window_offset, tcblock_rowid, tcblocktile_id, tcblock_offset, sparse_a_to_x_idx, block_count = prep
        use_balance = os.getenv("SPARSENE_DTC_BALANCE", "0") == "1"
        exeplan = os.getenv("SPARSENE_DTC_EXEPLAN", "float2_nonsplit")

        tile_b = _select_tile_b(int(dense.shape[1]), "SPARSENE_DTC_TILE_B")
        os.environ["DTC_TILE_B"] = str(tile_b)
        if self.variant == "strict_lb":
            os.environ["DTC_STRICT_LB_TILE_B"] = str(tile_b)
        dense_in, dense_out_cols = _pad_dense_to_tile(dense, tile_b)

        run_func = self._run_func_balance if use_balance else self._run_func_default
        run_func_name = self._run_func_balance_name if use_balance else self._run_func_default_name
        if run_func is None:
            return _fallback_or_raise(
                f"No callable DTC run function for variant={self.variant}",
                rowptr,
                colind,
                values,
                dense,
                num_rows,
            )
        try:
            if use_balance:
                out = run_func(
                    dense_in,
                    tcblock_rowid,
                    tcblocktile_id,
                    tcblock_offset,
                    sparse_a_to_x_idx,
                    int(num_rows),
                    exeplan,
                )
            elif run_func_name == "DTCSpMM_gcn":
                out = run_func(
                    dense_in,
                    row_window_offset,
                    tcblocktile_id,
                    tcblock_offset,
                    sparse_a_to_x_idx,
                    int(num_rows),
                    int(colind.numel()),
                )
            else:
                out = run_func(
                    dense_in,
                    row_window_offset,
                    tcblocktile_id,
                    tcblock_offset,
                    sparse_a_to_x_idx,
                    int(num_rows),
                    int(colind.numel()),
                    exeplan,
                )
        except Exception as exc:
            return _fallback_or_raise(
                f"DTC run failed for variant={self.variant}: {exc}",
                rowptr,
                colind,
                values,
                dense,
                num_rows,
            )

        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, torch.Tensor) and out.dim() == 2 and int(out.shape[1]) != dense_out_cols:
            out = out[:, :dense_out_cols].contiguous()
        return out

    def run(
        self,
        rowptr: Tensor,
        colind: Tensor,
        values: Tensor,
        dense: Tensor,
        num_rows: int,
        feat_dim: int,
        num_nodes_ori: int,
    ) -> Tensor:
        if self.variant not in {"base", "strict_lb", "multi_binding"}:
            _warn_once(f"Unknown DTC variant={self.variant}, fallback to base dispatch")
        return self._run_variant(rowptr, colind, values, dense, num_rows)


class SRBCRSBackend(BackendBase):
    """SR-BCRS bridge using src_fp32 testbed extension module.

    Expected source root:
      /workspace/sparsene/examples/src_fp32/sr_bcrs/testbed
    """

    def __init__(self, variant: str) -> None:
        self.variant = self._normalize_variant(variant)
        self._module = None
        self._cache: Dict[Tuple[int, int, int, int, int, str], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]] = {}
        self._i32_index_cache: Dict[Tuple[int, Tuple[int, ...], str, torch.dtype], Tensor] = {}
        self._f32_value_cache: Dict[Tuple[int, Tuple[int, ...], str, torch.dtype], Tensor] = {}
        self._preprocess_func: Optional[Callable[..., Any]] = None
        self._run_func_default: Optional[Callable[..., Any]] = None
        self._run_func_balance: Optional[Callable[..., Any]] = None
        self._source_root = Path(
            os.getenv(
                "SPARSENE_SRBCRS_SOURCE_ROOT",
                "/workspace/sparsene/examples/src_fp32/sr_bcrs/testbed",
            )
        ).resolve()
        self._module_name = os.getenv("SPARSENE_SRBCRS_MODULE_NAME", "SRBCRSSpMM").strip() or "SRBCRSSpMM"
        self._load_module()

    @staticmethod
    def _normalize_variant(variant: str) -> str:
        v = (variant or "base").strip().lower()
        if v in {"", "base", "default", "srbcrs", "sr_bcrs"}:
            return "base"
        if v in {"16x8", "16_8", "srbcrs_16x8", "sr_bcrs_16x8"}:
            return "16x8"
        if v in {"multi_bind", "multi-binding", "multi_binding", "multibind"}:
            return "multi_binding"
        if v in {"strict_lb", "strict-lb", "strictlb"}:
            return "strict_lb"
        if v in {
            "16x8_multi_bind",
            "16x8_multibind",
            "16x8_multi_binding",
            "multi_bind_16x8",
            "multi_binding_16x8",
            "multibind_16x8",
            "srbcrs_16x8_multi_bind",
            "sr_bcrs_16x8_multi_bind",
        }:
            return "16x8_multi_binding"
        if v in {
            "16x8_strict_lb",
            "16x8_strictlb",
            "strict_lb_16x8",
            "strictlb_16x8",
            "srbcrs_16x8_strict_lb",
            "sr_bcrs_16x8_strict_lb",
        }:
            return "16x8_strict_lb"
        return v

    @staticmethod
    def _variant_tile_shape(variant: str) -> Tuple[int, int]:
        if variant in {"16x8", "16x8_multi_binding", "16x8_strict_lb"}:
            return 16, 8
        return 32, 32

    @staticmethod
    def _iter_build_lib_dirs(source_root: Path) -> Iterable[Path]:
        build_dir = source_root / "build"
        if not build_dir.exists():
            return []
        out = []
        for child in build_dir.glob("lib*"):
            if child.is_dir():
                out.append(child)
        return out

    def _push_module_search_paths(self) -> None:
        candidates = []

        user_module_path = os.getenv("SPARSENE_SRBCRS_MODULE_PATH", "").strip()
        if user_module_path:
            candidates.append(Path(user_module_path))

        candidates.append(self._source_root)
        candidates.extend(self._iter_build_lib_dirs(self._source_root))

        for candidate in candidates:
            if candidate.exists():
                sys.path.insert(0, str(candidate))

    def _module_is_from_allowed_root(self, module: Any) -> bool:
        module_file = getattr(module, "__file__", "") or ""
        if not module_file:
            return False

        allow_non_src = os.getenv("SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32", "0") == "1"
        if allow_non_src:
            return True

        try:
            module_path = Path(module_file).resolve()
        except Exception:
            return False

        try:
            module_path.relative_to(self._source_root)
            return True
        except Exception:
            return False

    @staticmethod
    def _find_first_attr(module: Any, names: Iterable[str]) -> Optional[str]:
        for name in names:
            if hasattr(module, name):
                return name
        return None

    def _to_i32_cached(self, t: Tensor) -> Tensor:
        if t.dtype == torch.int32 and t.is_contiguous():
            return t
        key = (int(t.data_ptr()), tuple(t.shape), str(t.device), t.dtype)
        cached = self._i32_index_cache.get(key)
        if cached is not None:
            return cached
        converted = t.to(torch.int32).contiguous()
        self._i32_index_cache[key] = converted
        return converted

    def _to_f32_cached(self, t: Tensor) -> Tensor:
        if t.dtype == torch.float32 and t.is_contiguous():
            return t
        key = (int(t.data_ptr()), tuple(t.shape), str(t.device), t.dtype)
        cached = self._f32_value_cache.get(key)
        if cached is not None:
            return cached
        converted = t.to(torch.float32).contiguous()
        self._f32_value_cache[key] = converted
        return converted

    def _resolve_entrypoints(self) -> None:
        self._preprocess_func = None
        self._run_func_default = None
        self._run_func_balance = None
        if self._module is None:
            return

        preprocess_candidates: Dict[str, Tuple[str, ...]] = {
            "base": ("preprocess_gpu",),
            "16x8": ("preprocess_gpu_16x8", "preprocess_gpu"),
            "multi_binding": (
                "preprocess_gpu_multi_binding",
                "preprocess_gpu_multi_bind",
                "preprocess_gpu",
            ),
            "strict_lb": (
                "preprocess_gpu_strict_lb",
                "preprocess_gpu",
            ),
            "16x8_multi_binding": (
                "preprocess_gpu_16x8_multi_binding",
                "preprocess_gpu_16x8_multi_bind",
                "preprocess_gpu_multi_binding_16x8",
                "preprocess_gpu_multi_bind_16x8",
                "preprocess_gpu_16x8",
                "preprocess_gpu_multi_binding",
                "preprocess_gpu_multi_bind",
                "preprocess_gpu",
            ),
            "16x8_strict_lb": (
                "preprocess_gpu_16x8_strict_lb",
                "preprocess_gpu_strict_lb_16x8",
                "preprocess_gpu_16x8",
                "preprocess_gpu_strict_lb",
                "preprocess_gpu",
            ),
        }
        run_candidates: Dict[str, Tuple[str, ...]] = {
            "base": ("run_SRBCRS",),
            "16x8": ("run_SRBCRS_16x8", "run_SRBCRS"),
            "multi_binding": (
                "run_SRBCRS_multi_binding",
                "run_SRBCRS_multi_bind",
                "run_SRBCRS",
            ),
            "strict_lb": (
                "run_SRBCRS_strict_lb",
                "run_SRBCRS",
            ),
            "16x8_multi_binding": (
                "run_SRBCRS_16x8_multi_binding",
                "run_SRBCRS_16x8_multi_bind",
                "run_SRBCRS_multi_binding_16x8",
                "run_SRBCRS_multi_bind_16x8",
                "run_SRBCRS_16x8",
                "run_SRBCRS_multi_binding",
                "run_SRBCRS_multi_bind",
                "run_SRBCRS",
            ),
            "16x8_strict_lb": (
                "run_SRBCRS_16x8_strict_lb",
                "run_SRBCRS_strict_lb_16x8",
                "run_SRBCRS_16x8",
                "run_SRBCRS_strict_lb",
                "run_SRBCRS",
            ),
        }
        run_balance_candidates: Dict[str, Tuple[str, ...]] = {
            "base": ("run_SRBCRS_balance", "run_SRBCRS"),
            "16x8": ("run_SRBCRS_16x8_balance", "run_SRBCRS_16x8", "run_SRBCRS_balance", "run_SRBCRS"),
            "multi_binding": (
                "run_SRBCRS_multi_binding_balance",
                "run_SRBCRS_multi_bind_balance",
                "run_SRBCRS_multi_binding",
                "run_SRBCRS_multi_bind",
                "run_SRBCRS_balance",
                "run_SRBCRS",
            ),
            "strict_lb": (
                "run_SRBCRS_strict_lb_balance",
                "run_SRBCRS_strict_lb",
                "run_SRBCRS_balance",
                "run_SRBCRS",
            ),
            "16x8_multi_binding": (
                "run_SRBCRS_16x8_multi_binding_balance",
                "run_SRBCRS_16x8_multi_bind_balance",
                "run_SRBCRS_multi_binding_16x8_balance",
                "run_SRBCRS_multi_bind_16x8_balance",
                "run_SRBCRS_16x8_multi_binding",
                "run_SRBCRS_16x8_multi_bind",
                "run_SRBCRS_multi_binding_16x8",
                "run_SRBCRS_multi_bind_16x8",
                "run_SRBCRS_16x8_balance",
                "run_SRBCRS_16x8",
                "run_SRBCRS_multi_binding_balance",
                "run_SRBCRS_multi_bind_balance",
                "run_SRBCRS_multi_binding",
                "run_SRBCRS_multi_bind",
                "run_SRBCRS_balance",
                "run_SRBCRS",
            ),
            "16x8_strict_lb": (
                "run_SRBCRS_16x8_strict_lb_balance",
                "run_SRBCRS_strict_lb_16x8_balance",
                "run_SRBCRS_16x8_strict_lb",
                "run_SRBCRS_strict_lb_16x8",
                "run_SRBCRS_16x8_balance",
                "run_SRBCRS_16x8",
                "run_SRBCRS_strict_lb_balance",
                "run_SRBCRS_strict_lb",
                "run_SRBCRS_balance",
                "run_SRBCRS",
            ),
        }

        variant = self.variant if self.variant in preprocess_candidates else "base"
        preprocess_name = self._find_first_attr(self._module, preprocess_candidates[variant])
        if preprocess_name is not None:
            self._preprocess_func = getattr(self._module, preprocess_name)
        else:
            _warn_once(
                f"No preprocess function for SR-BCRS variant={self.variant} in module={self._module_name}"
            )

        run_name = self._find_first_attr(self._module, run_candidates[variant])
        if run_name is not None:
            self._run_func_default = getattr(self._module, run_name)

        run_balance_name = self._find_first_attr(self._module, run_balance_candidates[variant])
        if run_balance_name is not None:
            self._run_func_balance = getattr(self._module, run_balance_name)

    def _load_module(self) -> None:
        self._push_module_search_paths()

        try:
            module = importlib.import_module(self._module_name)
            if not self._module_is_from_allowed_root(module):
                module_file = getattr(module, "__file__", "unknown")
                _warn_once(
                    "Ignoring non-src_fp32 SR-BCRS module: "
                    f"{module_file}. Set SPARSENE_SRBCRS_ALLOW_NON_SRC_FP32=1 to override."
                )
                self._module = None
                return
            self._module = module
            self._resolve_entrypoints()
            return
        except Exception as exc:
            _warn_once(
                "src_fp32 SR-BCRS module import failed "
                f"(module={self._module_name}, root={self._source_root}): {exc}"
            )

        _warn_once("src_fp32 SR-BCRS extension unavailable, fallback to torch SpMM")

    def _get_preprocessed(
        self,
        rowptr: Tensor,
        colind: Tensor,
        values: Tensor,
        num_rows: int,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]]:
        if self._module is None:
            return None
        if not rowptr.is_cuda or not colind.is_cuda or not values.is_cuda:
            return None

        default_h, default_w = self._variant_tile_shape(self.variant)
        block_h = int(os.getenv("SPARSENE_SRBCRS_BLK_H", str(default_h)))
        block_w = int(os.getenv("SPARSENE_SRBCRS_BLK_W", str(default_w)))

        cache_key = (
            int(rowptr.data_ptr()),
            int(colind.data_ptr()),
            int(values.data_ptr()),
            int(num_rows),
            int(colind.numel()),
            f"{self.variant}:{block_h}x{block_w}",
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            with _nvtx_range("srbcrs.preprocess.cache_hit"):
                return cached

        row_windows = (num_rows + block_h - 1) // block_h
        edge_num = int(colind.numel())

        block_partition = torch.zeros(row_windows, dtype=torch.int32, device=rowptr.device)
        edge_to_column = torch.zeros(edge_num, dtype=torch.int32, device=rowptr.device)
        edge_to_row = torch.zeros(edge_num, dtype=torch.int32, device=rowptr.device)

        if self._preprocess_func is None:
            return None

        with _nvtx_range("srbcrs.preprocess.extension"):
            prep = self._preprocess_func(
                self._to_i32_cached(colind),
                self._to_i32_cached(rowptr),
                self._to_f32_cached(values),
                int(num_rows),
                int(block_h),
                int(block_w),
                block_partition,
                edge_to_column,
                edge_to_row,
            )
        self._cache[cache_key] = prep
        return prep

    def _run_variant(self, rowptr: Tensor, colind: Tensor, values: Tensor, dense: Tensor, num_rows: int) -> Tensor:
        with _nvtx_range(f"srbcrs.run.entry.{self.variant}"):
            prep = self._get_preprocessed(rowptr, colind, values, num_rows)
            if prep is None:
                with _nvtx_range("srbcrs.fallback.preprocess_none"):
                    return _fallback_or_raise(
                        f"SR-BCRS preprocess unavailable for variant={self.variant}",
                        rowptr,
                        colind,
                        values,
                        dense,
                        num_rows,
                    )

            row_window_offset, tcblock_rowid, tcblocktile_id, tcblock_offset, sparse_a_to_x_idx, block_count = prep
            use_balance = os.getenv("SPARSENE_SRBCRS_BALANCE", "0") == "1"
            exeplan = os.getenv("SPARSENE_SRBCRS_EXEPLAN", "")

            if os.getenv("SPARSENE_SRBCRS_FORCE_N64_ALIGN", "0") == "1" and "SPARSENE_SRBCRS_TILE_B" not in os.environ:
                os.environ["SPARSENE_SRBCRS_TILE_B"] = "64"

            tile_b = _select_tile_b(int(dense.shape[1]), "SPARSENE_SRBCRS_TILE_B")
            os.environ["SRBCRS_TILE_B"] = str(tile_b)
            with _nvtx_range(f"srbcrs.run.pad_tile{tile_b}"):
                dense_in, dense_out_cols = _pad_dense_to_tile(dense, tile_b)

            run_func = self._run_func_balance if use_balance else self._run_func_default
            if run_func is None:
                return _fallback_or_raise(
                    f"No callable SR-BCRS run function for variant={self.variant}",
                    rowptr,
                    colind,
                    values,
                    dense,
                    num_rows,
                )

            try:
                run_range = "srbcrs.run.balance" if use_balance else "srbcrs.run.default"
                with _nvtx_range(run_range):
                    if use_balance:
                        out = run_func(
                            dense_in,
                            row_window_offset,
                            tcblocktile_id,
                            tcblock_offset,
                            sparse_a_to_x_idx,
                            int(num_rows),
                            exeplan,
                        )
                    else:
                        out = run_func(
                            dense_in,
                            row_window_offset,
                            tcblocktile_id,
                            tcblock_offset,
                            sparse_a_to_x_idx,
                            int(num_rows),
                            int(colind.numel()),
                            exeplan,
                        )
            except Exception as exc:
                with _nvtx_range("srbcrs.fallback.exception"):
                    return _fallback_or_raise(
                        f"SR-BCRS run failed for variant={self.variant}: {exc}",
                        rowptr,
                        colind,
                        values,
                        dense,
                        num_rows,
                    )

            if isinstance(out, (tuple, list)):
                out = out[0]
            if isinstance(out, torch.Tensor) and out.dim() == 2 and int(out.shape[1]) != dense_out_cols:
                with _nvtx_range(f"srbcrs.run.unpad_tile{tile_b}"):
                    out = out[:, :dense_out_cols].contiguous()
            return out

    def run(
        self,
        rowptr: Tensor,
        colind: Tensor,
        values: Tensor,
        dense: Tensor,
        num_rows: int,
        feat_dim: int,
        num_nodes_ori: int,
    ) -> Tensor:
        if self.variant not in {
            "base",
            "16x8",
            "multi_binding",
            "strict_lb",
            "16x8_multi_binding",
            "16x8_strict_lb",
        }:
            _warn_once(f"Unknown SR-BCRS variant={self.variant}, fallback to base dispatch")
        return self._run_variant(rowptr, colind, values, dense, num_rows)

class FlashSparseBackend(BackendBase):
    """FlashSparse bridge using FlashSparse extension modules.

    Expected source root:
      /workspace/baselines/FlashSparse/FlashSparse
    """

    def __init__(self, variant: str) -> None:
        self.variant = self._normalize_variant(variant)
        self._spmm_module = None
        self._block_module = None
        self._run_func: Optional[Callable[..., Any]] = None
        self._preprocess_func: Optional[Callable[..., Any]] = None
        self._preprocess_kind = ""
        self._cache: Dict[
            Tuple[int, int, int, int, int, str, str],
            Tuple[
                weakref.ReferenceType[Tensor],
                weakref.ReferenceType[Tensor],
                weakref.ReferenceType[Tensor],
                Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, int],
            ],
        ] = {}
        self._i32_index_cache: Dict[int, Tuple[weakref.ReferenceType[Tensor], Tensor]] = {}
        self._f32_value_cache: Dict[int, Tuple[weakref.ReferenceType[Tensor], Tensor]] = {}
        self._cpu_i32_cache: Dict[int, Tuple[weakref.ReferenceType[Tensor], Tensor]] = {}
        self._cpu_f32_cache: Dict[int, Tuple[weakref.ReferenceType[Tensor], Tensor]] = {}
        self._padded_rowptr_cache: Dict[
            Tuple[int, int, int, str, torch.dtype],
            Tuple[weakref.ReferenceType[Tensor], Tensor],
        ] = {}
        self._source_root = Path(
            os.getenv(
                "SPARSENE_FLASHSPARSE_SOURCE_ROOT",
                "/workspace/baselines/FlashSparse/FlashSparse",
            )
        ).resolve()
        self._spmm_module_name = (
            os.getenv("SPARSENE_FLASHSPARSE_SPMM_MODULE_NAME", "FS_SpMM").strip() or "FS_SpMM"
        )
        self._block_module_name = (
            os.getenv("SPARSENE_FLASHSPARSE_BLOCK_MODULE_NAME", "FS_Block_gpu").strip() or "FS_Block_gpu"
        )
        self._block_module_fallback_name = (
            os.getenv("SPARSENE_FLASHSPARSE_BLOCK_FALLBACK_MODULE_NAME", "FS_Block").strip() or "FS_Block"
        )
        self._load_modules()

    @staticmethod
    def _normalize_variant(variant: str) -> str:
        v = (variant or "base").strip().lower()
        if v in {"", "base", "default", "tf32", "balance", "tf32_balance"}:
            return "tf32_balance"
        return v

    @staticmethod
    def _iter_build_lib_dirs(source_root: Path) -> Iterable[Path]:
        build_dir = source_root / "build"
        if not build_dir.exists():
            return []
        out = []
        for child in build_dir.glob("lib*"):
            if child.is_dir():
                out.append(child)
        return out

    def _push_module_search_paths(self) -> None:
        candidates = []

        user_module_path = os.getenv("SPARSENE_FLASHSPARSE_MODULE_PATH", "").strip()
        if user_module_path:
            candidates.append(Path(user_module_path))

        candidates.append(self._source_root)
        candidates.extend(self._iter_build_lib_dirs(self._source_root))

        for candidate in candidates:
            if candidate.exists():
                sys.path.insert(0, str(candidate))

    def _module_is_from_allowed_root(self, module: Any) -> bool:
        module_file = getattr(module, "__file__", "") or ""
        if not module_file:
            return False

        allow_non_src = os.getenv("SPARSENE_FLASHSPARSE_ALLOW_NON_DEFAULT_ROOT", "0") == "1"
        if allow_non_src:
            return True

        try:
            module_path = Path(module_file).resolve()
        except Exception:
            return False

        try:
            module_path.relative_to(self._source_root)
            return True
        except Exception:
            return False

    @staticmethod
    def _find_first_attr(module: Any, names: Iterable[str]) -> Optional[str]:
        for name in names:
            if hasattr(module, name):
                return name
        return None

    def _identity_cached(
        self,
        cache: Dict[int, Tuple[weakref.ReferenceType[Tensor], Tensor]],
        src: Tensor,
        make_value: Callable[[], Tensor],
    ) -> Tensor:
        key = id(src)
        entry = cache.get(key)
        if entry is not None:
            src_ref, cached = entry
            if src_ref() is src:
                return cached
        converted = make_value()
        cache[key] = (weakref.ref(src), converted)
        return converted

    def _to_i32_cached(self, t: Tensor) -> Tensor:
        if t.dtype == torch.int32 and t.is_contiguous():
            return t
        return self._identity_cached(
            self._i32_index_cache,
            t,
            lambda: t.to(torch.int32).contiguous(),
        )

    def _to_f32_cached(self, t: Tensor) -> Tensor:
        if t.dtype == torch.float32 and t.is_contiguous():
            return t
        return self._identity_cached(
            self._f32_value_cache,
            t,
            lambda: t.to(torch.float32).contiguous(),
        )

    def _to_cpu_i32_cached(self, t: Tensor) -> Tensor:
        return self._identity_cached(
            self._cpu_i32_cache,
            t,
            lambda: t.to(device="cpu", dtype=torch.int32).contiguous(),
        )

    def _to_cpu_f32_cached(self, t: Tensor) -> Tensor:
        return self._identity_cached(
            self._cpu_f32_cache,
            t,
            lambda: t.to(device="cpu", dtype=torch.float32).contiguous(),
        )

    def _load_modules(self) -> None:
        self._push_module_search_paths()

        try:
            spmm_module = importlib.import_module(self._spmm_module_name)
            if not self._module_is_from_allowed_root(spmm_module):
                module_file = getattr(spmm_module, "__file__", "unknown")
                _warn_once(
                    "Ignoring FlashSparse SpMM module from non-default root: "
                    f"{module_file}. Set SPARSENE_FLASHSPARSE_ALLOW_NON_DEFAULT_ROOT=1 to override."
                )
            else:
                self._spmm_module = spmm_module
        except Exception as exc:
            _warn_once(
                "FlashSparse SpMM module import failed "
                f"(module={self._spmm_module_name}, root={self._source_root}): {exc}"
            )

        block_candidates = [self._block_module_name]
        if self._block_module_fallback_name and self._block_module_fallback_name != self._block_module_name:
            block_candidates.append(self._block_module_fallback_name)

        for mod_name in block_candidates:
            try:
                mod = importlib.import_module(mod_name)
                if not self._module_is_from_allowed_root(mod):
                    module_file = getattr(mod, "__file__", "unknown")
                    _warn_once(
                        "Ignoring FlashSparse Block module from non-default root: "
                        f"{module_file}. Set SPARSENE_FLASHSPARSE_ALLOW_NON_DEFAULT_ROOT=1 to override."
                    )
                    continue
                self._block_module = mod
                break
            except Exception:
                continue

        if self._block_module is None:
            _warn_once(
                "FlashSparse block preprocess module import failed "
                f"(tried: {', '.join(block_candidates)})"
            )

        if self._spmm_module is not None:
            run_name = self._find_first_attr(self._spmm_module, ("forward_tf32_gnn",))
            if run_name is not None:
                self._run_func = getattr(self._spmm_module, run_name)

        if self._block_module is not None:
            preprocess_name = self._find_first_attr(
                self._block_module,
                ("preprocess_gpu_fs_balance", "blockProcess_tf32_balance"),
            )
            if preprocess_name is not None:
                self._preprocess_func = getattr(self._block_module, preprocess_name)
                if preprocess_name == "preprocess_gpu_fs_balance":
                    self._preprocess_kind = "gpu_balance"
                else:
                    self._preprocess_kind = "cpu_balance"

        if self._run_func is None or self._preprocess_func is None:
            _warn_once("FlashSparse extension entrypoints unavailable, fallback to torch SpMM")

    def _pad_rowptr(self, rowptr: Tensor, num_rows: int, padded_rows: int) -> Tensor:
        if padded_rows <= num_rows:
            return rowptr
        key = (id(rowptr), int(num_rows), int(padded_rows), str(rowptr.device), rowptr.dtype)
        cached = self._padded_rowptr_cache.get(key)
        if cached is not None:
            src_ref, padded_cached = cached
            if src_ref() is rowptr:
                return padded_cached

        extra = padded_rows - num_rows
        last_ptr = int(rowptr[-1].item())
        tail = torch.full((extra,), last_ptr, dtype=rowptr.dtype, device=rowptr.device)
        padded = torch.cat([rowptr, tail], dim=0).contiguous()
        self._padded_rowptr_cache[key] = (weakref.ref(rowptr), padded)
        return padded

    def _get_preprocessed(
        self,
        rowptr: Tensor,
        colind: Tensor,
        values: Tensor,
        num_rows: int,
        num_nodes_ori: int,
        device: torch.device,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int, int]]:
        if self._run_func is None or self._preprocess_func is None:
            return None
        if not rowptr.is_cuda or not colind.is_cuda or not values.is_cuda:
            return None

        align = max(int(os.getenv("SPARSENE_FLASHSPARSE_ROW_ALIGN", "8")), 1)
        window = int(os.getenv("SPARSENE_FLASHSPARSE_WINDOW", "8"))
        wide = int(os.getenv("SPARSENE_FLASHSPARSE_WIDE", "4"))
        part = int(os.getenv("SPARSENE_FLASHSPARSE_PART", "32"))

        padded_rows = ((num_rows + align - 1) // align) * align
        rowptr_padded = self._pad_rowptr(self._to_i32_cached(rowptr), num_rows, padded_rows)
        colind_i32 = self._to_i32_cached(colind)
        values_f32 = self._to_f32_cached(values)

        cache_key = (
            id(rowptr_padded),
            id(colind_i32),
            id(values_f32),
            int(padded_rows),
            int(num_nodes_ori),
            str(device),
            self._preprocess_kind,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            rowptr_ref, colind_ref, values_ref, prep_cached = cached
            if rowptr_ref() is rowptr_padded and colind_ref() is colind_i32 and values_ref() is values_f32:
                with _nvtx_range("flashsparse.preprocess.cache_hit"):
                    return prep_cached

        rowptr_cpu = self._to_cpu_i32_cached(rowptr_padded)
        colind_cpu = self._to_cpu_i32_cached(colind_i32)
        values_cpu = self._to_cpu_f32_cached(values_f32)

        try:
            if self._preprocess_kind == "gpu_balance":
                with _nvtx_range("flashsparse.preprocess.gpu_balance"):
                    prep = self._preprocess_func(
                        rowptr_cpu,
                        colind_cpu,
                        int(padded_rows),
                        int(colind_i32.numel()),
                        int(window),
                        int(wide),
                        int(part),
                    )
            elif self._preprocess_kind == "cpu_balance":
                with _nvtx_range("flashsparse.preprocess.cpu_balance"):
                    prep = self._preprocess_func(
                        rowptr_cpu,
                        colind_cpu,
                        values_cpu,
                        int(window),
                        int(wide),
                        int(part),
                    )
            else:
                return None
        except Exception as exc:
            _warn_once(f"FlashSparse preprocess failed: {exc}; fallback to torch SpMM")
            return None

        if not isinstance(prep, (tuple, list)) or len(prep) < 5:
            _warn_once("FlashSparse preprocess returned unexpected format; fallback to torch SpMM")
            return None

        rowptr_fs = prep[0].to(device=device, dtype=torch.int32).contiguous()
        colind_fs = prep[1].to(device=device, dtype=torch.int32).contiguous()
        values_fs = prep[2].to(device=device, dtype=torch.float32).contiguous()
        t_window = prep[3].to(device=device, dtype=torch.int32).contiguous()
        t_atomic = prep[4].to(device=device, dtype=torch.int32).contiguous()

        out = (rowptr_fs, colind_fs, values_fs, t_window, t_atomic, padded_rows, int(num_nodes_ori))
        self._cache[cache_key] = (
            weakref.ref(rowptr_padded),
            weakref.ref(colind_i32),
            weakref.ref(values_f32),
            out,
        )
        return out

    def run(
        self,
        rowptr: Tensor,
        colind: Tensor,
        values: Tensor,
        dense: Tensor,
        num_rows: int,
        feat_dim: int,
        num_nodes_ori: int,
    ) -> Tensor:
        with _nvtx_range(f"flashsparse.run.entry.{self.variant}"):
            prep = self._get_preprocessed(rowptr, colind, values, num_rows, num_nodes_ori, dense.device)
            if prep is None:
                with _nvtx_range("flashsparse.fallback.preprocess_none"):
                    return _fallback_or_raise(
                        "FlashSparse preprocess unavailable",
                        rowptr,
                        colind,
                        values,
                        dense,
                        num_rows,
                    )

            rowptr_fs, colind_fs, values_fs, t_window, t_atomic, padded_rows, out_rows = prep
            dense_f32 = self._to_f32_cached(dense)
            run_func = self._run_func
            if run_func is None:
                with _nvtx_range("flashsparse.fallback.runfunc_none"):
                    return _fallback_or_raise(
                        "FlashSparse run entrypoint missing",
                        rowptr,
                        colind,
                        values,
                        dense,
                        num_rows,
                    )
            try:
                with _nvtx_range("flashsparse.run.extension"):
                    out = run_func(
                        rowptr_fs,
                        colind_fs,
                        values_fs,
                        t_window,
                        t_atomic,
                        dense_f32,
                        int(padded_rows),
                        int(feat_dim),
                        int(out_rows),
                    )
            except Exception as exc:
                with _nvtx_range("flashsparse.fallback.exception"):
                    return _fallback_or_raise(
                        f"FlashSparse run failed: {exc}",
                        rowptr,
                        colind,
                        values,
                        dense,
                        num_rows,
                    )

            if isinstance(out, (tuple, list)):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                with _nvtx_range("flashsparse.fallback.non_tensor"):
                    return _fallback_or_raise(
                        "FlashSparse run returned non-tensor output",
                        rowptr,
                        colind,
                        values,
                        dense,
                        num_rows,
                    )
            if int(out.shape[0]) != int(num_nodes_ori):
                out = out[: int(num_nodes_ori)]
            return out


class PlaceholderBackend(BackendBase):
    """Placeholder backends for future operators.

    Supported names (placeholder): acc / bitbsr
    """

    def __init__(self, name: str, variant: str) -> None:
        self.name = name
        self.variant = variant

    def run(
        self,
        rowptr: Tensor,
        colind: Tensor,
        values: Tensor,
        dense: Tensor,
        num_rows: int,
        feat_dim: int,
        num_nodes_ori: int,
    ) -> Tensor:
        return _fallback_or_raise(
            f"Backend {self.name}:{self.variant} is placeholder and has no kernel implementation",
            rowptr,
            colind,
            values,
            dense,
            num_rows,
        )


class BackendRouter:
    def __init__(self) -> None:
        self._instances: Dict[BackendConfig, BackendBase] = {}

    def _build(self, cfg: BackendConfig) -> BackendBase:
        if cfg.name == "torch":
            return TorchRefBackend()
        if cfg.name == "dtc":
            return DTCBackend(cfg.variant)
        if cfg.name == "sr_bcrs":
            return SRBCRSBackend(cfg.variant)
        if cfg.name == "flashsparse":
            return FlashSparseBackend(cfg.variant)
        if cfg.name in {"acc", "bitbsr"}:
            return PlaceholderBackend(cfg.name, cfg.variant)
        _warn_once(f"Unknown backend={cfg.name}, fallback to torch")
        return TorchRefBackend()

    def get(self, cfg: BackendConfig) -> BackendBase:
        inst = self._instances.get(cfg)
        if inst is None:
            inst = self._build(cfg)
            self._instances[cfg] = inst
        return inst


_ROUTER = BackendRouter()


def reset_backend_state() -> None:
    """Reset router instances to avoid cross-run stale preprocess caches.

    Some extension backends are sensitive to long-lived process state when
    multiple datasets are evaluated sequentially in one process.
    """

    global _ROUTER
    _ROUTER = BackendRouter()


def _resolve_config() -> BackendConfig:
    # Primary selector for extension path.
    # Examples:
    #   SPARSENE_SPMM_BACKEND=dtc SPARSENE_DTC_VARIANT=base
    #   SPARSENE_SPMM_BACKEND=dtc SPARSENE_DTC_VARIANT=strict_lb
    #   SPARSENE_SPMM_BACKEND=acc
    name = os.getenv("SPARSENE_SPMM_BACKEND", "dtc").strip().lower()
    variant = os.getenv("SPARSENE_SPMM_VARIANT", "").strip().lower()

    if name == "dtc":
        dtc_variant = os.getenv("SPARSENE_DTC_VARIANT", "base").strip().lower()
        return BackendConfig(name="dtc", variant=dtc_variant or "base")

    if name == "sr_bcrs":
        srbcrs_variant = os.getenv("SPARSENE_SRBCRS_VARIANT", variant or "base").strip().lower()
        return BackendConfig(name="sr_bcrs", variant=srbcrs_variant or "base")

    if name == "flashsparse":
        flash_variant = os.getenv("SPARSENE_FLASHSPARSE_VARIANT", variant or "tf32_balance").strip().lower()
        return BackendConfig(name="flashsparse", variant=flash_variant or "tf32_balance")

    return BackendConfig(name=name or "torch", variant=variant or "base")


def spmm_external(
    rowptr: Tensor,
    colind: Tensor,
    values: Tensor,
    dense: Tensor,
    num_rows: int,
    feat_dim: int,
    num_nodes_ori: int,
) -> Tensor:
    cfg = _resolve_config()
    with _nvtx_range(f"spmm.dispatch.{cfg.name}.{cfg.variant}"):
        backend = _ROUTER.get(cfg)
        return backend.run(rowptr, colind, values, dense, num_rows, feat_dim, num_nodes_ori)


# Explicit DTC entry for --external-function.
def dtc_spmm(
    rowptr: Tensor,
    colind: Tensor,
    values: Tensor,
    dense: Tensor,
    num_rows: int,
    feat_dim: int,
    num_nodes_ori: int,
) -> Tensor:
    cfg = BackendConfig(name="dtc", variant=os.getenv("SPARSENE_DTC_VARIANT", "base").strip().lower())
    backend = _ROUTER.get(cfg)
    return backend.run(rowptr, colind, values, dense, num_rows, feat_dim, num_nodes_ori)


def srbcrs_spmm(
    rowptr: Tensor,
    colind: Tensor,
    values: Tensor,
    dense: Tensor,
    num_rows: int,
    feat_dim: int,
    num_nodes_ori: int,
) -> Tensor:
    cfg = BackendConfig(
        name="sr_bcrs",
        variant=os.getenv("SPARSENE_SRBCRS_VARIANT", os.getenv("SPARSENE_SPMM_VARIANT", "base")).strip().lower(),
    )
    backend = _ROUTER.get(cfg)
    return backend.run(rowptr, colind, values, dense, num_rows, feat_dim, num_nodes_ori)


def flashsparse_spmm(
    rowptr: Tensor,
    colind: Tensor,
    values: Tensor,
    dense: Tensor,
    num_rows: int,
    feat_dim: int,
    num_nodes_ori: int,
) -> Tensor:
    cfg = BackendConfig(
        name="flashsparse",
        variant=os.getenv("SPARSENE_FLASHSPARSE_VARIANT", "tf32_balance").strip().lower() or "tf32_balance",
    )
    backend = _ROUTER.get(cfg)
    return backend.run(rowptr, colind, values, dense, num_rows, feat_dim, num_nodes_ori)
