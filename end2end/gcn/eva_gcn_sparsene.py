#!/usr/bin/env python3
"""GCN end-to-end benchmark with a replaceable SpMM backend.

This script follows the FlashSparse end2end workflow style:
- loop over datasets and hidden dimensions
- train for a fixed number of epochs
- report per-case training time to CSV

Default backend uses torch CSR SpMM so the pipeline can run immediately.
To plug in a custom kernel, pass --backend external plus module/function.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import importlib
import os
import subprocess
import sys
import time
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor

_DGL_CUDA_READY_CACHE: Dict[str, bool] = {}


@dataclass
class GraphData:
    name: str
    num_nodes: int
    num_edges: int
    rowptr: Tensor
    colind: Tensor
    values: Tensor
    rowptr_t: Tensor
    colind_t: Tensor
    values_t: Tensor
    edge_index: Tensor
    framework_graph: Optional[Any]
    x: Tensor
    y: Tensor


@dataclass
class RunTiming:
    total_sec: float
    forward_sec: float
    backward_sec: float
    zero_grad_sec: float
    step_sec: float


@dataclass
class TrainTimingSummary:
    train_time_sec_mean: float
    train_time_sec_std: float
    time_per_epoch_ms_mean: float
    time_per_epoch_ms_std: float
    forward_ms_per_epoch_mean: float
    backward_ms_per_epoch_mean: float
    zero_grad_ms_per_epoch_mean: float
    step_ms_per_epoch_mean: float
    preprocess_sec_excluded: float


def parse_int_list(spec: str) -> List[int]:
    values: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"Empty integer list: {spec}")
    return values


def parse_str_list(spec: str) -> List[str]:
    values = [x.strip() for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Empty string list: {spec}")
    return values


def resolve_dataset_paths(dataset_dir: str, datasets: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in datasets:
        if item.endswith(".npz"):
            if os.path.isabs(item):
                path = item
            else:
                path = os.path.join(dataset_dir, item)
            name = os.path.splitext(os.path.basename(item))[0]
        else:
            name = item
            path = os.path.join(dataset_dir, f"{item}.npz")
        if os.path.exists(path):
            pairs.append((name, path))
        else:
            print(f"[WARN] Skip missing dataset: {path}")
    return pairs


def _extract_graph_arrays(graph: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray, int]:
    if "src_li" in graph and "dst_li" in graph:
        src = graph["src_li"]
        dst = graph["dst_li"]
    elif "edge_index" in graph:
        edge_index = graph["edge_index"]
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        src = edge_index[0]
        dst = edge_index[1]
    elif "edge_index_new" in graph:
        edge_index = graph["edge_index_new"]
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index_new must have shape [2, E]")
        src = edge_index[0]
        dst = edge_index[1]
    else:
        raise KeyError("Dataset must contain src_li/dst_li or edge_index")

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)

    if "num_nodes_src" in graph:
        num_nodes = int(graph["num_nodes_src"])  # follows FlashSparse convention
    elif "num_nodes" in graph:
        num_nodes = int(graph["num_nodes"])
    else:
        num_nodes = int(max(src.max(initial=0), dst.max(initial=0)) + 1)

    return src, dst, num_nodes


def load_graph_data(
    dataset_name: str,
    dataset_path: str,
    feature_dim: int,
    num_classes: int,
    device: torch.device,
    seed: int,
) -> GraphData:
    g = np.load(dataset_path)
    src, dst, num_nodes = _extract_graph_arrays(g)
    num_edges = int(src.shape[0])

    indices = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    values = torch.ones(num_edges, dtype=torch.float32)

    a_coo = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    a_csr = a_coo.to_sparse_csr()
    a_t_csr = a_csr.transpose(0, 1).to_sparse_csr()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn((num_nodes, feature_dim), generator=gen, dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_nodes,), generator=gen, dtype=torch.long)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=device)

    return GraphData(
        name=dataset_name,
        num_nodes=num_nodes,
        num_edges=num_edges,
        rowptr=a_csr.crow_indices().to(device),
        colind=a_csr.col_indices().to(device),
        values=a_csr.values().to(device),
        rowptr_t=a_t_csr.crow_indices().to(device),
        colind_t=a_t_csr.col_indices().to(device),
        values_t=a_t_csr.values().to(device),
        edge_index=edge_index,
        framework_graph=None,
        x=x.to(device),
        y=y.to(device),
    )


class SpMMBackend:
    def __init__(self, name: str, external_fn: Optional[Callable[..., Tensor]] = None) -> None:
        self.name = name
        self.external_fn = external_fn
        # Cache converted indices by source tensor identity to avoid stale hits
        # when allocator/data_ptr values are reused across datasets.
        self._i32_index_cache: Dict[int, Tuple[weakref.ReferenceType[Tensor], Tensor]] = {}

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

    @staticmethod
    def build(backend: str, module_name: str, function_name: str) -> "SpMMBackend":
        if backend == "torch":
            return SpMMBackend(name="torch")

        if backend != "external":
            raise ValueError(f"Unsupported backend: {backend}")

        if not module_name or not function_name:
            raise ValueError("External backend requires --external-module and --external-function")

        module = importlib.import_module(module_name)
        fn = getattr(module, function_name)
        return SpMMBackend(name=f"external:{module_name}.{function_name}", external_fn=fn)

    def spmm(self, rowptr: Tensor, colind: Tensor, values: Tensor, dense: Tensor) -> Tensor:
        n_rows = int(rowptr.numel() - 1)
        if self.external_fn is None:
            csr = torch.sparse_csr_tensor(
                crow_indices=rowptr,
                col_indices=colind,
                values=values,
                size=(n_rows, n_rows),
                device=dense.device,
            )
            return torch.sparse.mm(csr, dense)

        # Expected signature for custom backend:
        # fn(rowptr, colind, values, dense, num_rows, feat_dim, num_nodes_ori)
        out = self.external_fn(
            self._to_i32_cached(rowptr),
            self._to_i32_cached(colind),
            values,
            dense,
            n_rows,
            int(dense.shape[1]),
            n_rows,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise TypeError("External SpMM backend must return a torch.Tensor")
        return out


class ReplaceableSpMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rowptr: Tensor,
        colind: Tensor,
        values: Tensor,
        rowptr_t: Tensor,
        colind_t: Tensor,
        values_t: Tensor,
        dense: Tensor,
        backend: SpMMBackend,
    ) -> Tensor:
        ctx.backend = backend
        ctx.save_for_backward(rowptr_t, colind_t, values_t)
        return backend.spmm(rowptr, colind, values, dense)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor):
        grad_output = grad_outputs[0]
        rowptr_t, colind_t, values_t = ctx.saved_tensors
        grad_dense = ctx.backend.spmm(rowptr_t, colind_t, values_t, grad_output)
        return None, None, None, None, None, None, grad_dense, None


class ReplaceableGCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((in_dim, out_dim), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor, data: GraphData, backend: SpMMBackend) -> Tensor:
        x = x @ self.weight
        out = ReplaceableSpMMFunction.apply(
            data.rowptr,
            data.colind,
            data.values,
            data.rowptr_t,
            data.colind_t,
            data.values_t,
            x,
            backend,
        )
        return cast(Tensor, out)


class ReplaceableGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.conv_in = ReplaceableGCNConv(in_dim, hidden_dim)
        self.hidden = nn.ModuleList(
            [ReplaceableGCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        )
        self.conv_out = ReplaceableGCNConv(hidden_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, data: GraphData, backend: SpMMBackend) -> Tensor:
        x = F.relu(self.conv_in(data.x, data, backend))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.hidden:
            x = F.relu(conv(x, data, backend))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_out(x, data, backend)
        return F.log_softmax(x, dim=1)


class PyGGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        try:
            from torch_geometric.nn import GCNConv  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "PyG backend requires torch_geometric. "
                "Install it in the current Python environment first."
            ) from exc

        self.conv_in = GCNConv(
            in_dim,
            hidden_dim,
            add_self_loops=False,
            normalize=False,
            bias=False,
        )
        self.hidden = nn.ModuleList(
            [
                GCNConv(
                    hidden_dim,
                    hidden_dim,
                    add_self_loops=False,
                    normalize=False,
                    bias=False,
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.conv_out = GCNConv(
            hidden_dim,
            out_dim,
            add_self_loops=False,
            normalize=False,
            bias=False,
        )
        self.dropout = float(dropout)

    def forward(self, data: GraphData, backend: SpMMBackend) -> Tensor:
        _ = backend
        edge_index = data.edge_index
        x = F.relu(self.conv_in(data.x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.hidden:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_out(x, edge_index)
        return F.log_softmax(x, dim=1)


class DGLGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        try:
            import dgl.nn.pytorch as dglnn  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "DGL backend requires dgl. Install it in the current Python environment first."
            ) from exc

        self.conv_in = dglnn.GraphConv(
            in_dim,
            hidden_dim,
            norm="none",
            weight=True,
            bias=False,
            allow_zero_in_degree=True,
        )
        self.hidden = nn.ModuleList(
            [
                dglnn.GraphConv(
                    hidden_dim,
                    hidden_dim,
                    norm="none",
                    weight=True,
                    bias=False,
                    allow_zero_in_degree=True,
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.conv_out = dglnn.GraphConv(
            hidden_dim,
            out_dim,
            norm="none",
            weight=True,
            bias=False,
            allow_zero_in_degree=True,
        )
        self.dropout = float(dropout)

    def forward(self, data: GraphData, backend: SpMMBackend) -> Tensor:
        _ = backend
        if data.framework_graph is None:
            raise RuntimeError("DGL graph is not prepared")
        graph = data.framework_graph
        x = F.relu(self.conv_in(graph, data.x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.hidden:
            x = F.relu(conv(graph, x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_out(graph, x)
        return F.log_softmax(x, dim=1)


def _prepare_framework_graph(data: GraphData, backend_name: str, device: torch.device) -> None:
    if backend_name != "dgl":
        return

    try:
        import dgl  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "DGL backend requires dgl. Install it in the current Python environment first."
        ) from exc

    if device.type == "cuda":
        cache_key = str(device)
        if not _DGL_CUDA_READY_CACHE.get(cache_key, False):
            try:
                import dgl.nn.pytorch as dglnn  # type: ignore

                probe_src = torch.tensor([0], dtype=torch.int64, device=device)
                probe_dst = torch.tensor([0], dtype=torch.int64, device=device)
                probe_graph = dgl.graph((probe_src, probe_dst), num_nodes=1, device=device)
                probe_x = torch.ones((1, 1), dtype=torch.float32, device=device)
                probe_conv = dglnn.GraphConv(
                    1,
                    1,
                    norm="none",
                    weight=False,
                    bias=False,
                    allow_zero_in_degree=True,
                ).to(device)
                _ = probe_conv(probe_graph, probe_x)
                _DGL_CUDA_READY_CACHE[cache_key] = True
            except Exception as exc:
                raise RuntimeError(
                    "DGL CUDA backend is unavailable in the current environment. "
                    "Current dgl is likely CPU-only. Install a CUDA-enabled DGL build, "
                    "or run with --device cpu for DGL backend."
                ) from exc

    src = data.edge_index[0].to(torch.int64)
    dst = data.edge_index[1].to(torch.int64)
    data.framework_graph = dgl.graph((src, dst), num_nodes=data.num_nodes, device=device)


def _build_model(
    backend_name: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    if backend_name in {"torch", "external"}:
        return ReplaceableGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if backend_name == "pyg":
        return PyGGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if backend_name == "dgl":
        return DGLGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model backend: {backend_name}")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _clone_model_state(model: nn.Module) -> Dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _prewarm_backend(
    backend: SpMMBackend,
    data: GraphData,
    iters: int,
) -> float:
    if backend.external_fn is None or iters <= 0:
        return 0.0

    probe_dense = torch.zeros((data.num_nodes, 1), dtype=torch.float32, device=data.x.device)
    _sync_if_cuda(data.x.device)
    start = time.perf_counter()
    with _nvtx_range(f"bench.prewarm.{backend.name}"):
        with torch.no_grad():
            for _ in range(iters):
                with _nvtx_range("bench.prewarm.spmm.forward"):
                    _ = backend.spmm(data.rowptr, data.colind, data.values, probe_dense)
                with _nvtx_range("bench.prewarm.spmm.transpose"):
                    _ = backend.spmm(data.rowptr_t, data.colind_t, data.values_t, probe_dense)
    _sync_if_cuda(data.x.device)
    end = time.perf_counter()
    return end - start


def _run_one_epoch(
    model: nn.Module,
    data: GraphData,
    backend: SpMMBackend,
    optimizer: torch.optim.Optimizer,
    segment_timing: bool,
) -> RunTiming:
    model.train()
    device = data.x.device

    if device.type == "cuda" and segment_timing:
        with _nvtx_range(f"train.epoch.{backend.name}"):
            _sync_if_cuda(device)
            total_start = time.perf_counter()

            f_start = torch.cuda.Event(enable_timing=True)
            f_end = torch.cuda.Event(enable_timing=True)
            z_start = torch.cuda.Event(enable_timing=True)
            z_end = torch.cuda.Event(enable_timing=True)
            b_start = torch.cuda.Event(enable_timing=True)
            b_end = torch.cuda.Event(enable_timing=True)
            s_start = torch.cuda.Event(enable_timing=True)
            s_end = torch.cuda.Event(enable_timing=True)

            with _nvtx_range("train.forward"):
                f_start.record()
                logits = model(data, backend)
                loss = F.nll_loss(logits, data.y)
                f_end.record()

            with _nvtx_range("train.zero_grad"):
                z_start.record()
                optimizer.zero_grad(set_to_none=True)
                z_end.record()

            with _nvtx_range("train.backward"):
                b_start.record()
                loss.backward()
                b_end.record()

            with _nvtx_range("train.step"):
                s_start.record()
                optimizer.step()
                s_end.record()

            _sync_if_cuda(device)
            total_end = time.perf_counter()

        return RunTiming(
            total_sec=total_end - total_start,
            forward_sec=f_start.elapsed_time(f_end) / 1000.0,
            backward_sec=b_start.elapsed_time(b_end) / 1000.0,
            zero_grad_sec=z_start.elapsed_time(z_end) / 1000.0,
            step_sec=s_start.elapsed_time(s_end) / 1000.0,
        )

    with _nvtx_range(f"train.epoch.{backend.name}"):
        _sync_if_cuda(device)
        total_start = time.perf_counter()

        with _nvtx_range("train.forward"):
            f_start = time.perf_counter()
            logits = model(data, backend)
            loss = F.nll_loss(logits, data.y)
            _sync_if_cuda(device)
            f_end = time.perf_counter()

        with _nvtx_range("train.zero_grad"):
            z_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            _sync_if_cuda(device)
            z_end = time.perf_counter()

        with _nvtx_range("train.backward"):
            b_start = time.perf_counter()
            loss.backward()
            _sync_if_cuda(device)
            b_end = time.perf_counter()

        with _nvtx_range("train.step"):
            s_start = time.perf_counter()
            optimizer.step()
            _sync_if_cuda(device)
            s_end = time.perf_counter()

        total_end = time.perf_counter()

    if not segment_timing:
        return RunTiming(total_sec=total_end - total_start,
                         forward_sec=0.0,
                         backward_sec=0.0,
                         zero_grad_sec=0.0,
                         step_sec=0.0)

    return RunTiming(
        total_sec=total_end - total_start,
        forward_sec=f_end - f_start,
        backward_sec=b_end - b_start,
        zero_grad_sec=z_end - z_start,
        step_sec=s_end - s_start,
    )


def train_timed(
    model: nn.Module,
    data: GraphData,
    backend: SpMMBackend,
    epochs: int,
    warmup_epochs: int,
    repeat_runs: int,
    segment_timing: bool,
    exclude_preprocess: bool,
    backend_warmup_iters: int,
    lr: float,
    weight_decay: float,
) -> TrainTimingSummary:
    if repeat_runs <= 0:
        raise ValueError("repeat_runs must be >= 1")

    preprocess_sec = 0.0
    if exclude_preprocess:
        preprocess_sec = _prewarm_backend(backend, data, backend_warmup_iters)

    base_state = _clone_model_state(model)
    run_timings: List[RunTiming] = []

    for _ in range(repeat_runs):
        model.load_state_dict(base_state, strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(warmup_epochs):
            _run_one_epoch(model, data, backend, optimizer, segment_timing=False)

        total_sec = 0.0
        forward_sec = 0.0
        backward_sec = 0.0
        zero_grad_sec = 0.0
        step_sec = 0.0

        for _ in range(epochs):
            timing = _run_one_epoch(model, data, backend, optimizer, segment_timing=segment_timing)
            total_sec += timing.total_sec
            forward_sec += timing.forward_sec
            backward_sec += timing.backward_sec
            zero_grad_sec += timing.zero_grad_sec
            step_sec += timing.step_sec

        run_timings.append(
            RunTiming(
                total_sec=total_sec,
                forward_sec=forward_sec,
                backward_sec=backward_sec,
                zero_grad_sec=zero_grad_sec,
                step_sec=step_sec,
            )
        )

    denom = float(max(epochs, 1))
    total_arr = np.array([x.total_sec for x in run_timings], dtype=np.float64)
    epoch_ms_arr = total_arr * 1000.0 / denom

    if segment_timing:
        fwd_ms_arr = np.array([x.forward_sec for x in run_timings], dtype=np.float64) * 1000.0 / denom
        bwd_ms_arr = np.array([x.backward_sec for x in run_timings], dtype=np.float64) * 1000.0 / denom
        zgd_ms_arr = np.array([x.zero_grad_sec for x in run_timings], dtype=np.float64) * 1000.0 / denom
        stp_ms_arr = np.array([x.step_sec for x in run_timings], dtype=np.float64) * 1000.0 / denom
    else:
        fwd_ms_arr = np.zeros_like(epoch_ms_arr)
        bwd_ms_arr = np.zeros_like(epoch_ms_arr)
        zgd_ms_arr = np.zeros_like(epoch_ms_arr)
        stp_ms_arr = np.zeros_like(epoch_ms_arr)

    return TrainTimingSummary(
        train_time_sec_mean=float(total_arr.mean()),
        train_time_sec_std=float(total_arr.std()),
        time_per_epoch_ms_mean=float(epoch_ms_arr.mean()),
        time_per_epoch_ms_std=float(epoch_ms_arr.std()),
        forward_ms_per_epoch_mean=float(fwd_ms_arr.mean()),
        backward_ms_per_epoch_mean=float(bwd_ms_arr.mean()),
        zero_grad_ms_per_epoch_mean=float(zgd_ms_arr.mean()),
        step_ms_per_epoch_mean=float(stp_ms_arr.mean()),
        preprocess_sec_excluded=preprocess_sec,
    )


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _sanitize_token(token: str) -> str:
    out = []
    for ch in token:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _should_isolate_flashsparse(args: argparse.Namespace, dataset_count: int) -> bool:
    if args._internal_no_isolate:
        return False
    if dataset_count <= 1:
        return False
    if os.getenv("SPARSENE_FLASHSPARSE_ISOLATE", "1") != "1":
        return False
    if args.backend != "external":
        return False

    fn = (args.external_function or "").strip().lower()
    env_backend = os.getenv("SPARSENE_SPMM_BACKEND", "").strip().lower()

    if fn == "flashsparse_spmm":
        return True
    if fn == "spmm_external" and env_backend == "flashsparse":
        return True
    if env_backend == "flashsparse":
        return True
    return False


def _build_child_cli_args(
    args: argparse.Namespace,
    dataset: str,
    output_csv: str,
) -> List[str]:
    return [
        "--dataset-dir",
        args.dataset_dir,
        "--datasets",
        dataset,
        "--hidden-list",
        args.hidden_list,
        "--layer-list",
        args.layer_list,
        "--epochs",
        str(args.epochs),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--repeat-runs",
        str(args.repeat_runs),
        "--segment-timing",
        str(args.segment_timing),
        "--exclude-preprocess",
        str(args.exclude_preprocess),
        "--backend-warmup-iters",
        str(args.backend_warmup_iters),
        "--feature-dim",
        str(args.feature_dim),
        "--num-classes",
        str(args.num_classes),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--backend",
        args.backend,
        "--external-module",
        args.external_module,
        "--external-function",
        args.external_function,
        "--output-csv",
        output_csv,
        "--_internal-no-isolate",
    ]


def _run_flashsparse_isolated(
    args: argparse.Namespace,
    dataset_names: Sequence[str],
) -> None:
    ensure_parent_dir(args.output_csv)

    script_path = os.path.abspath(__file__)
    output_root, output_ext = os.path.splitext(args.output_csv)
    if not output_ext:
        output_ext = ".csv"

    child_csvs: List[str] = []
    for idx, dataset_name in enumerate(dataset_names):
        safe_ds = _sanitize_token(dataset_name)
        child_csv = f"{output_root}.__iso__.{idx:03d}.{safe_ds}{output_ext}"
        child_csvs.append(child_csv)

        cmd = [
            sys.executable,
            script_path,
            *_build_child_cli_args(args=args, dataset=dataset_name, output_csv=child_csv),
        ]

        print(f"[ISO-RUN] {dataset_name}")
        subprocess.run(cmd, check=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        wrote_header = False

        for child_csv in child_csvs:
            with open(child_csv, "r", newline="", encoding="utf-8") as in_f:
                rows = list(csv.reader(in_f))
            if not rows:
                continue
            if not wrote_header:
                writer.writerow(rows[0])
                wrote_header = True
            for row in rows[1:]:
                writer.writerow(row)

        if not wrote_header:
            raise RuntimeError("No CSV rows produced in isolated FlashSparse runs")

    for child_csv in child_csvs:
        try:
            os.remove(child_csv)
        except OSError:
            pass

    print(f"[ISO-MERGE] wrote {args.output_csv} from {len(child_csvs)} isolated runs")


def main() -> None:
    parser = argparse.ArgumentParser(description="GCN end2end benchmark with replaceable SpMM")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory with .npz graphs")
    parser.add_argument(
        "--datasets",
        type=str,
        default="DD,OVCAR-8H,Yeast,YeastH,ddi,protein,reddit,web-BerkStan",
        help="Comma-separated dataset names or .npz filenames",
    )
    parser.add_argument("--hidden-list", type=str, default="64,128,256")
    parser.add_argument("--layer-list", type=str, default="3,6")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument(
        "--segment-timing",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable segmented timing (forward/backward/optimizer).",
    )
    parser.add_argument(
        "--exclude-preprocess",
        type=int,
        default=1,
        choices=[0, 1],
        help="Prewarm backend preprocess before timing so it is excluded from train_time.",
    )
    parser.add_argument(
        "--backend-warmup-iters",
        type=int,
        default=2,
        help="Number of backend prewarm SpMM iterations (forward+transpose) before timing.",
    )
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "external", "pyg", "dgl"],
        default="torch",
    )
    parser.add_argument("--external-module", type=str, default="")
    parser.add_argument("--external-function", type=str, default="")
    parser.add_argument("--_internal-no-isolate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="/workspace/sparsene/end2end/result/gcn_e2e.csv",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    device = torch.device(args.device)

    hidden_list = parse_int_list(args.hidden_list)
    layer_list = parse_int_list(args.layer_list)
    datasets = parse_str_list(args.datasets)

    dataset_paths = resolve_dataset_paths(args.dataset_dir, datasets)
    if not dataset_paths:
        raise FileNotFoundError("No dataset files found. Check --dataset-dir and --datasets")

    if _should_isolate_flashsparse(args=args, dataset_count=len(dataset_paths)):
        _run_flashsparse_isolated(args=args, dataset_names=[name for name, _ in dataset_paths])
        return

    if args.backend in {"torch", "external"}:
        backend = SpMMBackend.build(args.backend, args.external_module, args.external_function)
    else:
        backend = SpMMBackend(name=args.backend)
    enable_external_reset = os.getenv("SPARSENE_EXTERNAL_RESET_PER_CASE", "0") == "1"
    external_reset_fn: Optional[Callable[[], None]] = None
    if args.backend == "external" and args.external_module:
        try:
            ext_module = importlib.import_module(args.external_module)
            maybe_reset = getattr(ext_module, "reset_backend_state", None)
            if callable(maybe_reset):
                external_reset_fn = cast(Callable[[], None], maybe_reset)
        except Exception as exc:
            print(f"[WARN] Failed to resolve external reset hook: {exc}")
    ensure_parent_dir(args.output_csv)

    header = [
        "dataset",
        "num_nodes",
        "num_edges",
        "layers",
        "hidden",
        "feature_dim",
        "epochs",
        "warmup_epochs",
        "backend",
        "device",
        "repeat_runs",
        "segment_timing",
        "exclude_preprocess",
        "backend_warmup_iters",
        "preprocess_sec_excluded",
        "train_time_sec",
        "train_time_std_sec",
        "time_per_epoch_ms",
        "time_per_epoch_std_ms",
        "forward_ms_per_epoch",
        "backward_ms_per_epoch",
        "zero_grad_ms_per_epoch",
        "step_ms_per_epoch",
    ]

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for hidden in hidden_list:
        for layers in layer_list:
            for dataset_name, dataset_path in dataset_paths:
                if enable_external_reset and external_reset_fn is not None:
                    external_reset_fn()
                data = load_graph_data(
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    feature_dim=args.feature_dim,
                    num_classes=args.num_classes,
                    device=device,
                    seed=args.seed,
                )
                _prepare_framework_graph(data=data, backend_name=args.backend, device=device)

                model = _build_model(
                    backend_name=args.backend,
                    in_dim=args.feature_dim,
                    hidden_dim=hidden,
                    out_dim=args.num_classes,
                    num_layers=layers,
                    dropout=args.dropout,
                ).to(device)

                summary = train_timed(
                    model=model,
                    data=data,
                    backend=backend,
                    epochs=args.epochs,
                    warmup_epochs=args.warmup_epochs,
                    repeat_runs=args.repeat_runs,
                    segment_timing=bool(args.segment_timing),
                    exclude_preprocess=bool(args.exclude_preprocess),
                    backend_warmup_iters=args.backend_warmup_iters,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )

                print(
                    f"[OK] {dataset_name} L={layers} H={hidden} "
                    f"time={summary.train_time_sec_mean:.4f}s "
                    f"(+/- {summary.train_time_sec_std:.4f}s, "
                    f"{summary.time_per_epoch_ms_mean:.4f} ms/epoch) "
                    f"seg[fwd={summary.forward_ms_per_epoch_mean:.3f}, "
                    f"bwd={summary.backward_ms_per_epoch_mean:.3f}, "
                    f"step={summary.step_ms_per_epoch_mean:.3f}] ms/epoch "
                    f"preprocess_excluded={summary.preprocess_sec_excluded:.4f}s"
                )

                row = [
                    dataset_name,
                    data.num_nodes,
                    data.num_edges,
                    layers,
                    hidden,
                    args.feature_dim,
                    args.epochs,
                    args.warmup_epochs,
                    backend.name,
                    args.device,
                    args.repeat_runs,
                    args.segment_timing,
                    args.exclude_preprocess,
                    args.backend_warmup_iters,
                    f"{summary.preprocess_sec_excluded:.6f}",
                    f"{summary.train_time_sec_mean:.6f}",
                    f"{summary.train_time_sec_std:.6f}",
                    f"{summary.time_per_epoch_ms_mean:.6f}",
                    f"{summary.time_per_epoch_ms_std:.6f}",
                    f"{summary.forward_ms_per_epoch_mean:.6f}",
                    f"{summary.backward_ms_per_epoch_mean:.6f}",
                    f"{summary.zero_grad_ms_per_epoch_mean:.6f}",
                    f"{summary.step_ms_per_epoch_mean:.6f}",
                ]
                with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)


if __name__ == "__main__":
    main()
