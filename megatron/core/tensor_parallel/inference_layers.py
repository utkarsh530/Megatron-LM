# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from typing import Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TELinear,
    TERowParallelLinear,
)
from megatron.core.inference.communication.torch_symm_triton import (
    are_tensors_nvls_eligible,
    fused_multimem_rs_add_norm_ag,
    multimem_all_gather,
    multimem_reduce_scatter,
)
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
from megatron.core.inference.quantization.utils import mm_mxfp8
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none

try:
    import transformer_engine.pytorch.cpp_extensions as tex
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.distributed import (
        gather_along_first_dim,
        reduce_scatter_along_first_dim,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def _nrl_gd_record(t: torch.Tensor) -> None:
    """PATCH(NRL_GDUMP): capture-safe in-graph recorder (device ops only).
    Records t[0:16] rows for [M>=56, 2048] tensors + per-record kv-length meta.
    STOP-AT-FULL (12288 slots); counter zeroed at serving step 1 by
    dynamic_context; saver thread + SIGTERM/atexit persist per-rank file."""
    import os as _os_gd
    if not _os_gd.environ.get("NRL_GDUMP", ""):
        return
    if t.dim() != 2 or t.size(-1) != 2048 or t.size(0) < 56:
        return
    _g = globals()
    if "_NRL_GD_BUF" not in _g:
        _g["_NRL_GD_BUF"] = torch.zeros(12288, 16, 2048, dtype=torch.float32, device=t.device)
        _g["_NRL_GD_META"] = torch.zeros(12288, 16, dtype=torch.int32, device=t.device)
        _g["_NRL_GD_CNT"] = torch.zeros(1, dtype=torch.int64, device=t.device)
        _g["_NRL_GD_ON"] = torch.ones(1, dtype=torch.int64, device=t.device)
        import atexit as _ax, signal as _sig
        def _gd_save(*_sig_args):
            try:
                torch.save({"buf": _g["_NRL_GD_BUF"].cpu(), "cnt": int(_g["_NRL_GD_CNT"].item()),
                            "meta": _g["_NRL_GD_META"].cpu()},
                           _os_gd.environ["NRL_GDUMP"] + "/gdump_rank" + _os_gd.environ.get("RANK", "0") + ".pt")
            except Exception:
                pass
            if _sig_args:
                raise SystemExit(0)
        _ax.register(_gd_save)
        try:
            _sig.signal(_sig.SIGTERM, _gd_save)
        except Exception:
            pass
        import threading as _th, time as _tm
        def _gd_loop():
            _tick = 0
            while True:
                _tm.sleep(0.2)
                _tick += 1
                if _tick % 25:
                    continue
                try:
                    _tmp = (_os_gd.environ["NRL_GDUMP"] + "/gdump_rank"
                            + _os_gd.environ.get("RANK", "0") + ".pt.tmp")
                    torch.save({"buf": _g["_NRL_GD_BUF"].cpu(),
                                "cnt": int(_g["_NRL_GD_CNT"].item()),
                                "meta": _g["_NRL_GD_META"].cpu()}, _tmp)
                    _os_gd.replace(_tmp, _os_gd.environ["NRL_GDUMP"] + "/gdump_rank"
                                   + _os_gd.environ.get("RANK", "0") + ".pt")
                except Exception:
                    pass
        _th.Thread(target=_gd_loop, daemon=True).start()
    _slot = _g["_NRL_GD_CNT"].clamp(max=12287)
    _x0 = t[0:16].float()
    _gate48 = torch.ones(1, dtype=torch.int64, device=t.device)
    for _lag in (96, 384):  # 96 records/step now (norm-in + proj-out per layer)
        _ref = _g["_NRL_GD_BUF"].index_select(0, (_g["_NRL_GD_CNT"] - _lag).clamp(min=0, max=12287))
        _gate48 = _gate48 * (_x0 != _ref[0]).any().to(torch.int64).reshape(1)
    _gate_full = (_g["_NRL_GD_CNT"] < 12287).to(torch.int64)
    _g["_NRL_GD_BUF"].index_copy_(0, _slot, _x0.unsqueeze(0))
    import builtins as _bi_gd
    _kv = getattr(_bi_gd, "_NRL_KV_GPU", None)
    if _kv is not None:
        _g["_NRL_GD_META"].index_copy_(0, _slot, _kv[:16].to(torch.int32).unsqueeze(0))
    _g["_NRL_GD_CNT"].add_(_g["_NRL_GD_ON"] * _gate48 * _gate_full)


def _te_rms_norm_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float):
    if not globals().get("_NRL_RMSK_BANNER"):
        globals()["_NRL_RMSK_BANNER"] = True
        import sys as _sys_rk
        print(f"[NRL_RMSK] _te_rms_norm_kernel FIRST CALL x={tuple(x.shape)}",
              file=_sys_rk.stderr, flush=True)
    x_shape = x.shape
    x = x.view(-1, x.size(-1))
    # PATCH(NRL_GDUMP): capture-safe in-graph ring buffer — records the first 16 rows
    # of every 2048-dim norm input AND every 2048-out dense GEMM output (proj) so
    # graph REPLAY steps are observable. Stream interleave per step:
    # [norm-in L0, proj-out L0, norm-in L1, proj-out L1, ...] = 96 records/step.
    _nrl_gd_record(x)
    import os as _os_gd
    if False and _os_gd.environ.get("NRL_GDUMP", ""):
        # ALL ranks (requests are DP-sharded across EP ranks; victim rank unknown)
        _g = globals()
        if "_NRL_GD_BUF" not in _g:
            _g["_NRL_GD_BUF"] = torch.zeros(12288, 16, 2048, dtype=torch.float32, device=x.device)
            # per-record request kv lengths (device-side read of the live gpu
            # bookkeeping buffer -> valid under graph replay): alignment key
            _g["_NRL_GD_META"] = torch.zeros(12288, 16, dtype=torch.int32, device=x.device)
            _g["_NRL_GD_CNT"] = torch.zeros(1, dtype=torch.int64, device=x.device)
            _g["_NRL_GD_ON"] = torch.ones(1, dtype=torch.int64, device=x.device)
            import atexit as _ax, signal as _sig
            def _gd_save(*_sig_args):
                try:
                    torch.save({"buf": _g["_NRL_GD_BUF"].cpu(), "cnt": int(_g["_NRL_GD_CNT"].item()),
                                "meta": _g["_NRL_GD_META"].cpu()},
                               _os_gd.environ["NRL_GDUMP"] + "/gdump_rank" + _os_gd.environ.get("RANK", "0") + ".pt")
                except Exception:
                    pass
                if _sig_args:
                    raise SystemExit(0)
            _ax.register(_gd_save)
            try:
                _sig.signal(_sig.SIGTERM, _gd_save)
            except Exception:
                pass
            # signal registration fails off-main-thread: periodic saver thread as
            # the reliable path (atomic tmp+rename; final idle snapshot is clean)
            import threading as _th, time as _tm
            def _gd_loop():
                _tick = 0
                while True:
                    _tm.sleep(0.2)
                    _tick += 1
                    try:
                        _mark = _os_gd.path.join(_os_gd.path.dirname(_os_gd.environ["NRL_GDUMP"]), "ipqdone_status")
                        if (_os_gd.path.exists("/tmp/nrl_gdfreeze") or _os_gd.path.exists(_mark)) and int(_g["_NRL_GD_ON"].item()):
                            _g["_NRL_GD_ON"].zero_()
                            print("[NRL_GDUMP] FROZEN (generation done)", flush=True)
                    except Exception:
                        pass
                    if _tick % 25:
                        continue
                    try:
                        _tmp = (_os_gd.environ["NRL_GDUMP"] + "/gdump_rank"
                                + _os_gd.environ.get("RANK", "0") + ".pt.tmp")
                        torch.save({"buf": _g["_NRL_GD_BUF"].cpu(),
                                    "cnt": int(_g["_NRL_GD_CNT"].item()),
                                    "meta": _g["_NRL_GD_META"].cpu()}, _tmp)
                        _os_gd.replace(_tmp, _os_gd.environ["NRL_GDUMP"] + "/gdump_rank" + _os_gd.environ.get("RANK", "0") + ".pt")
                    except Exception:
                        pass
            _th.Thread(target=_gd_loop, daemon=True).start()
        # STOP-AT-FULL (not a ring): the onset window is EARLY (steps ~5-25);
        # wrapping would let the 250+ later steps overwrite it. When full, keep
        # rewriting the last slot.
        _slot = _g["_NRL_GD_CNT"].clamp(max=12287)
        # lag-48 self-similarity gate: idle dummy steps repeat the SAME 48 per-layer
        # rows bitwise every step — if this row equals the row recorded one step ago
        # at the same call position, don't advance (at most one trailing dummy step
        # is retained). Real decode rows change every step. Pure device ops.
        _x0 = x[0:16].float()  # LIVE rows (decode rows 0..bs-1 live at the front)
        _gate48 = torch.ones(1, dtype=torch.int64, device=x.device)
        for _lag in (48, 192):
            _ref = _g["_NRL_GD_BUF"].index_select(0, (_g["_NRL_GD_CNT"] - _lag).clamp(min=0, max=12287))
            _gate48 = _gate48 * (_x0 != _ref[0]).any().to(torch.int64).reshape(1)
        _gate_full = (_g["_NRL_GD_CNT"] < 12287).to(torch.int64)
        _g["_NRL_GD_BUF"].index_copy_(0, _slot, _x0.unsqueeze(0))
        import builtins as _bi_gd
        _kv = getattr(_bi_gd, "_NRL_KV_GPU", None)
        if _kv is not None:
            _g["_NRL_GD_META"].index_copy_(0, _slot, _kv[:16].to(torch.int32).unsqueeze(0))
        _g["_NRL_GD_CNT"].add_(_g["_NRL_GD_ON"] * _gate48 * _gate_full)
        import sys as _sys_gd
        if torch.cuda.is_current_stream_capturing():
            if _g.get("_NRL_GD_CAP_N", 0) < 3:
                _g["_NRL_GD_CAP_N"] = _g.get("_NRL_GD_CAP_N", 0) + 1
                print(f"[GDUMP] block CAPTURED into graph (shape={tuple(x.shape)})",
                      file=_sys_gd.stderr, flush=True)
        elif _g.get("_NRL_GD_EAG_N", 0) < 8:
            _g["_NRL_GD_EAG_N"] = _g.get("_NRL_GD_EAG_N", 0) + 1
            print(f"[GDUMP] eager hit shape={tuple(x.shape)} "
                  f"cnt={int(_g['_NRL_GD_CNT'].item())}", file=_sys_gd.stderr, flush=True)
    # PATCH(NRL_OPDUMP prenorm @ function chokepoint): capture raw pre-norm rows of the
    # first prefill-like calls (embedding output at layer 1).
    import os as _os_pn
    _dd = _os_pn.environ.get("NRL_OPDUMP", "")
    if _dd and _os_pn.environ.get("RANK", "0") == "0" and not torch.cuda.is_current_stream_capturing():
        _st = globals().setdefault("_NRL_PRENORM_ST", {"n": 0})
        if (_st["n"] < 8 and x.shape[0] >= 25 and not bool(torch.isnan(x[:16]).any())
                and not bool((x[0] == x[1]).all())):
            torch.save(x[:32].detach().float().cpu().clone(), _dd + f"/prenorm_A_{_st['n']}.pt")
            _st["n"] += 1
    out, _, _ = tex.rmsnorm_fwd(
        x, weight, eps, None, None, TE_DType[x.dtype], 16, False  # sm-margin  # zero centered gamma
    )
    out = out.view(*x_shape[:-1], -1)
    return out.to(x.dtype)


def _apply_linear(
    x: torch.Tensor,
    weight: Union[torch.Tensor, MXFP8Tensor],
    config: TransformerConfig,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Helper to apply either MXFP8 or standard GEMM based on the configuration.
    """
    kwargs = {"out": out} if out is not None else {}
    if isinstance(weight, MXFP8Tensor):
        return mm_mxfp8(x, weight, **kwargs)
    # PATCH(NRL_DET_INFOPT BI dense GEMM): route the inference dense GEMMs through the
    # batch-invariant Triton matmul DIRECTLY (call-site, no dispatcher monkeypatch —
    # 2312829 proved the aten override never intercepts these calls in the server).
    import os as _os_det
    if _os_det.environ.get("NRL_DET_INFOPT", "0") == "1" and x.dtype == torch.bfloat16:
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            BatchInvariantTEGemmFn,
        )
        _gemm_impl = _os_det.environ.get("NRL_BI_GEMM_IMPL", "triton")
        if not globals().get("_NRL_BI_APPLY_BANNER"):
            globals()["_NRL_BI_APPLY_BANNER"] = True
            import sys as _sys_d
            print(f"[NRL_DET_INFOPT] _apply_linear -> BI matmul ACTIVE impl={_gemm_impl} (x={tuple(x.shape)})",
                  file=_sys_d.stderr, flush=True)
        x2 = x.reshape(-1, x.shape[-1])
        if _gemm_impl in ("native", "aten_native"):
            # PATCH(NRL_BI_GEMM_IMPL, GEMM Path-C probe): native cuBLASLt TN GEMM
            # (F.linear) under the launcher's workspace pins, paired with the scoring
            # side's native fall-through in batch_invariant_kernels. Empirical question:
            # is native cuBLASLt already row-invariant across the engine/scoring M
            # classes on this workload (like the NVLS combine turned out to be)?
            # RESULT (2329415): NON-ZERO (~1.5e-3, 51% exact) — but F.linear (aten)
            # is a DIFFERENT cuBLASLt entry point than scoring's TE general_gemm,
            # so this arm can't separate entry-point mismatch from M-variance.
            r = torch.nn.functional.linear(x2, weight)
        elif _gemm_impl == "te_native":
            # PATCH(NRL_BI_GEMM_IMPL=te_native): call the EXACT same TE general_gemm
            # (nvjet) the scoring forward runs — same entry point, same TN layout,
            # same workspace-pinned algo space. If nvjet is row-invariant across the
            # engine/scoring M classes (golden det-TE evidence says it was), this is
            # exact AND drops the Triton tax. If te_cpp.general_gemm is BI-patched in
            # this process, the wrapper's native fall-through routes to the ORIG.
            # This TE's general_gemm(A, B, out_dtype=..., layout=...) fetches its own
            # cuBLAS workspace internally (get_cublas_workspace) — no workspace arg.
            import transformer_engine.pytorch.cpp_extensions as _te_cpp_gi
            r = _te_cpp_gi.general_gemm(weight, x2, out_dtype=x2.dtype, layout="TN")[0]
        else:
            # EXACT same fn+layout the TE-side BI patch runs (TN, contiguous transposed weight):
            # a bare mm(x, w.t()) on the strided view produced 1-ulp diffs on near-zero elements
            # (bisect 2313053: pos-12 QKV seed).
            r = BatchInvariantTEGemmFn.apply(weight, x2, None, x2.dtype, "TN")
        # PATCH(NRL_GDUMP): record 2048-out GEMM outputs (= attention proj-out; the
        # only dense GEMM with out-features 2048) for the attn-vs-MoE flip bracket
        _nrl_gd_record(r.reshape(-1, weight.shape[0]))
        r = r.reshape(*x.shape[:-1], weight.shape[0])
        # PATCH(NRL_OPDUMP): in-vivo op-boundary dump for the intra-layer bisect.
        # Records (call_idx, x rows, out rows) for the first 200 calls on RANK 0.
        _dd = _os_det.environ.get("NRL_OPDUMP", "")
        if _dd and _os_det.environ.get("RANK", "0") == "0" and not torch.cuda.is_current_stream_capturing():
            _st = globals().setdefault("_NRL_OPDUMP_ST", {"buf": [], "n": 0, "wfp": False})
            if not _os_det.path.exists(_dd + "/TRIGGER"):
                return (out.copy_(r) or out) if out is not None else r
            if not _st["wfp"]:
                _st["wfp"] = True
                torch.save({"w_sum": weight.float().abs().sum().item(),
                            "w_head": weight[:2, :8].detach().float().cpu().clone(),
                            "w_shape": tuple(weight.shape)}, _dd + "/wfp_A.pt")
                # full weight for offline recompute triangulation (once, ~40MB)
                torch.save(weight.detach().cpu().clone(), _dd + "/wfull_A.pt")
            _pfx = x2[:16]
            _pf_like = (x2.shape[0] >= 25 and not bool(torch.isnan(_pfx).any())
                        and not bool((x2[0] == x2[1]).all()))
            if _st["n"] < 2000 and _pf_like:
                _rows = list(range(25))
                _r2 = r.reshape(-1, r.shape[-1])
                _st["buf"].append((_st["n"], tuple(x.shape),
                                   x2[_rows].detach().float().cpu().clone(),
                                   _r2[_rows].detach().float().cpu().clone()))
                _st["n"] += 1
                if _st["n"] % 64 == 0:
                    torch.save(_st["buf"], _dd + "/opdump_A.pt")
        if out is not None:
            out.copy_(r)
            return out
        return r
    return torch.matmul(x, weight.t(), **kwargs)


class InferenceLinear(TELinear):
    """Inference optimized version of TELinear."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        tp_comm_buffer_name: Optional[str] = None,
        is_expert: bool = False,
        symmetric_ar_type: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        name: str | None = None,
    ):
        assert HAVE_TE, "--transformer-impl=inference_optimized requires transformer engine"
        super().__init__(
            input_size,
            output_size,
            parallel_mode=parallel_mode,
            config=config,
            init_method=init_method,
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            is_expert=is_expert,
            symmetric_ar_type=symmetric_ar_type,
            tp_group=tp_group,
            name=name,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Forward pass."""
        if self.training:
            return super().forward(x)

        x = _apply_linear(x, self.weight, self.config)
        return x, None


class InferenceLayerNormColumnParallelLinear(TELayerNormColumnParallelLinear):
    """
    Inference optimized version of TELayerNormColumnParallelLinear.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        stride: int = 1,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        name: str | None = None,
    ):
        assert HAVE_TE, "--transformer-impl=inference_optimized requires transformer engine"
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            stride=stride,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
            name=name,
        )
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.tp_size = dist.get_world_size(self.tp_group)

        assert (
            output_size % self.tp_size == 0
        ), f"output_size ({output_size}) must be divisible by tp_size ({self.tp_size})"

        self.eps = config.layernorm_epsilon

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--transformer-impl=inference_optimized requires --sequence-parallel"

        self.triton_nvls_kernels_allowed = not config.inference_disable_triton_nvls_kernels

        # Boolean to be toggled externally for skipping norm and all-gather.
        # This is used when enabling fused reduce-scatter + add + rms-norm + all-gather
        # in tensor parallelism. In this case, the preceeding RowParallelLinear layer
        # has already applied the rms-norm and all-gather.
        self.skip_norm_and_all_gather = False

    def _maybe_allocate_symmetric_buffer(self, x: torch.Tensor):
        """
        Attempt to allocate symmetric memory buffer for all-gather.
        """
        symm_mem_buffer_dims = list(x.size())
        symm_mem_buffer_dims[0] *= self.tp_size
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=self.tp_group)
        symm_mem_buffer = buf.maybe_get_tensor(symm_mem_buffer_dims, dtype=x.dtype)
        return symm_mem_buffer

    def _all_gather(self, x: torch.Tensor, symm_mem_buffer: dict) -> None:
        """
        Attempt an NVLS all-gather into symmetric memory. If not possible,
        revert to torch dist (NCCL) all-gather.
        """
        if self.tp_size == 1:
            return x

        # Check input only: if input is 16-byte divisible, the output
        # (world_size * input) is too.
        can_use_nvls = (
            self.triton_nvls_kernels_allowed
            and are_tensors_nvls_eligible(x)
            and symm_mem_buffer["handle"] is not None
        )
        if can_use_nvls:
            # do multimem all gather
            multimem_all_gather(symm_mem_buffer["tensor"], x, symm_mem_buffer["handle"])
            return symm_mem_buffer["tensor"]
        else:
            # revert to torch dist (NCCL) all gather
            x, _ = gather_along_first_dim(x, process_group=self.tp_group)
            return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Forward pass.
        """
        # Necessary conditions to ensure we are executing the fused rs-add-rmsnorm-ag
        # in the preceeding RowParallelLinear layer.
        # 1. skip_norm_and_all_gather is True
        # 2. tp_size > 1
        # 3. enough symmetric memory is available - if available it already has the output

        if self.training:
            return super().forward(x)

        if self.tp_size == 1:
            # PATCH(NRL_OPDUMP prenorm): capture the RAW layer input (embedding output at
            # layer 1) for the first prefill-like call — tests whether the pos-12 seed
            # predates the norm (embedding-weight bit diff across load paths).
            import os as _os_pn
            _dd = _os_pn.environ.get("NRL_OPDUMP", "")
            if _dd and not torch.cuda.is_current_stream_capturing() and _os_pn.environ.get("RANK", "0") == "0":
                _st = globals().setdefault("_NRL_PRENORM_ST", {"n": 0})
                x2v = x.reshape(-1, x.shape[-1])
                if (_st["n"] < 8 and x2v.shape[0] >= 25 and not bool(torch.isnan(x2v[:16]).any())
                        and not bool((x2v[0] == x2v[1]).all())):
                    torch.save(x2v[:32].detach().float().cpu().clone(),
                               _dd + f"/prenorm_A_{_st['n']}.pt")
                    _st["n"] += 1
            x = _te_rms_norm_kernel(x=x, weight=self.layer_norm_weight, eps=self.eps)
            x = _apply_linear(x, self.weight, self.config)
            return x, None

        symm_mem_buffer = self._maybe_allocate_symmetric_buffer(x)
        is_in_fused_mode = (
            self.skip_norm_and_all_gather
            and self.tp_size > 1
            and symm_mem_buffer["handle"] is not None
        )
        if is_in_fused_mode:
            x = symm_mem_buffer["tensor"]
        else:
            x = _te_rms_norm_kernel(x=x, weight=self.layer_norm_weight, eps=self.eps)
            x = self._all_gather(x, symm_mem_buffer)

        x = _apply_linear(x, self.weight, self.config)

        return x, None


class InferenceColumnParallelLinear(TEColumnParallelLinear):
    """
    Inference optimized version of TEColumnParallelLinear.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        stride: int = 1,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        name: str | None = None,
    ):
        assert HAVE_TE, "--transformer-impl=inference_optimized requires transformer engine"
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            stride=stride,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
            name=name,
        )
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.tp_size = dist.get_world_size(self.tp_group)

        assert (
            output_size % self.tp_size == 0
        ), f"output_size ({output_size}) must be divisible by tp_size ({self.tp_size})"

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--transformer-impl=inference_optimized requires --sequence-parallel"

        self.triton_nvls_kernels_allowed = not config.inference_disable_triton_nvls_kernels

    def _maybe_allocate_symmetric_buffer(self, x: torch.Tensor):
        """
        Attempt to allocate symmetric memory buffer for all-gather.
        """
        symm_mem_buffer_dims = list(x.size())
        symm_mem_buffer_dims[0] *= self.tp_size
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=self.tp_group)
        symm_mem_buffer = buf.maybe_get_tensor(symm_mem_buffer_dims, dtype=x.dtype)
        return symm_mem_buffer

    def _all_gather(self, x: torch.Tensor, symm_mem_buffer: dict) -> None:
        """
        Attempt an NVLS all-gather into symmetric memory. If not possible,
        revert to torch dist (NCCL) all-gather.
        """
        if self.tp_size == 1:
            return x

        can_use_nvls = (
            self.triton_nvls_kernels_allowed
            and are_tensors_nvls_eligible(x)
            and symm_mem_buffer["handle"] is not None
        )
        if can_use_nvls:
            multimem_all_gather(symm_mem_buffer["tensor"], x, symm_mem_buffer["handle"])
            return symm_mem_buffer["tensor"]
        else:
            x, _ = gather_along_first_dim(x, process_group=self.tp_group)
            return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Forward pass.
        """
        if self.training:
            return super().forward(x)

        if self.tp_size == 1:
            x = _apply_linear(x, self.weight, self.config)
            return x, None

        symm_mem_buffer = self._maybe_allocate_symmetric_buffer(x)
        x = self._all_gather(x, symm_mem_buffer)
        x = _apply_linear(x, self.weight, self.config)

        return x, None


class InferenceRowParallelLinear(TERowParallelLinear):
    """
    Inference optimized version of TERowParallelLinear.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        name: str | None = None,
    ):
        assert HAVE_TE, "--transformer-impl=inference_optimized requires transformer engine"
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            tp_group=tp_group,
            name=name,
        )
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        self.tp_size = dist.get_world_size(self.tp_group)
        assert (
            input_size % self.tp_size == 0
        ), f"input_size ({input_size}) must be divisible by tp_size ({self.tp_size})"

        if self.tp_size > 1:
            assert (
                config.sequence_parallel
            ), "--transformer-impl=inference_optimized requires --sequence-parallel"

        self.triton_nvls_kernels_allowed = not getattr(
            config, 'inference_disable_triton_nvls_kernels', False
        )

        # Placeholder for next layer norm weights for fused
        # reduce-scatter + add + rms-norm + all-gather
        self.next_layer_norm_weights = None
        self.config = config

    def _matmul_reduce_scatter(self, x, residual=None):
        """
        Multiplies x by the weight matrix and performs a reduce-scatter.
        It will first try to write the matmul output to symmetric memory
        and perform an NVLS multicast reduce-scatter. If that is not possible,
        it will revert to torch.dist (NCCL) reduce-scatter.
        """
        use_mxfp8 = isinstance(self.weight, MXFP8Tensor)
        symm_mem_buffer_dims = list(x.size())
        if use_mxfp8:
            # Remove seq_len dimension for MXFP8 (mm_mxfp8 squeezes internally)
            del symm_mem_buffer_dims[1]
        symm_mem_buffer_dims[-1] = self.weight.size(0)
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=self.tp_group)
        symm_mem_buffer = buf.maybe_get_tensor(symm_mem_buffer_dims, dtype=x.dtype)

        # RS requires bf16 (hardware multimem reduce is bf16-only).
        # Check the matmul output shape: if it is NVLS-eligible, the RS output
        # (world_size times smaller on dim 0) is too.
        can_use_nvls = (
            self.triton_nvls_kernels_allowed
            and x.dtype == torch.bfloat16
            and are_tensors_nvls_eligible(x)
            and symm_mem_buffer["handle"] is not None
        )

        if can_use_nvls:
            # Write output of matmul directly onto the symmetric memory buffer

            x = _apply_linear(x, self.weight, self.config, out=symm_mem_buffer["tensor"])

            # perform nvls reduce-scatter
            if self.next_layer_norm_weights is None:
                output_dims = list(x.size())
                output_dims[0] = x.size(0) // self.tp_size
                output = torch.empty(output_dims, dtype=x.dtype, device=x.device)
                multimem_reduce_scatter(output, x, symm_mem_buffer["handle"])
                return output
            else:
                assert hasattr(self, "residual"), (
                    "For fused reduce-scatter + add + rms-norm + all-gather, "
                    "residual must be set via _set_residual()"
                )
                residual = self.residual
                fused_multimem_rs_add_norm_ag(
                    residual,
                    symm_mem_buffer["tensor"],
                    symm_mem_buffer["handle"],
                    residual,
                    self.next_layer_norm_weights,
                    self.config.layernorm_epsilon,
                )
                # 1. Residual has the output of the reduce-scatter + residual add
                #    Care must be taken in the model definition, so as to not apply the
                #    residual again.
                # 2. The output of the full reduce-scatter + add + rms-norm + all-gather is
                #    written into symm_mem_buffer["tensor"] and will be accessible there.
                return residual
        else:
            # revert to torch dist (NCCL) reduce-scatter
            x = _apply_linear(x, self.weight, self.config)
            x, _ = reduce_scatter_along_first_dim(x, tp_group=self.tp_group)
        return x

    def _set_next_layer_norm_weights(self, weights: torch.Tensor):
        """
        Set next layer norm weights for fused reduce-scatter + add + rms-norm + all-gather.
        """
        self.next_layer_norm_weights = weights

    def _set_residual(self, residual: torch.Tensor):
        """
        Set residual for fused reduce-scatter + add + rms-norm + all-gather.
        """
        self.residual = residual

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, None]:
        """
        Forward pass.
        """
        if self.training:
            return super().forward(x)

        if self.tp_size == 1:
            x = _apply_linear(x, self.weight, self.config)
            return x, None
        else:
            x = self._matmul_reduce_scatter(x)
            return x, None


def inference_all_gather_from_tensor_model_parallel_region(
    x: torch.Tensor, tp_group: torch.distributed.ProcessGroup, config: TransformerConfig
) -> torch.Tensor:
    """NVLS-optimized all-gather along the last dimension, with NCCL fallback.

    Replaces `gather_from_tensor_model_parallel_region` in inference paths
    where autograd is not needed and NVLS symmetric-memory is available.

    The NVLS path performs a flat all-gather into symmetric memory (concatenating
    along dim-0), then rearranges the result to the last dimension — the same
    semantics as `_gather_along_last_dim` but using hardware multicast when
    possible.
    """
    tp_size = dist.get_world_size(tp_group)
    if tp_size == 1:
        return x

    triton_nvls_kernels_allowed = not getattr(
        config, 'inference_disable_triton_nvls_kernels', False
    )

    if triton_nvls_kernels_allowed and SymmetricMemoryManager.is_initialized("tp"):
        ag_buffer_dims = list(x.size())
        ag_buffer_dims[0] *= tp_size
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=tp_group)
        symm_mem_buffer = buf.maybe_get_tensor(ag_buffer_dims, dtype=x.dtype)

        if are_tensors_nvls_eligible(x) and symm_mem_buffer["handle"] is not None:
            multimem_all_gather(symm_mem_buffer["tensor"], x, symm_mem_buffer["handle"])
            tensor_list = symm_mem_buffer["tensor"].chunk(tp_size, dim=0)
            return torch.cat(tensor_list, dim=-1).contiguous()

    return gather_from_tensor_model_parallel_region(x, group=tp_group)


def inference_reduce_scatter_to_sequence_parallel_region(
    x: torch.Tensor, tp_group: torch.distributed.ProcessGroup, config: TransformerConfig
) -> torch.Tensor:
    """NVLS-optimized reduce-scatter along the first dimension, with NCCL fallback.

    Replaces `reduce_scatter_to_sequence_parallel_region` in inference paths
    where autograd is not needed and NVLS symmetric-memory is available.
    """
    # TODO(ksanthanam): Refactor InferenceRowParallelLinear._matmul_reduce_scatter
    # to use this function for its non-fused NVLS reduce-scatter path.
    tp_size = dist.get_world_size(tp_group)
    if tp_size == 1:
        return x

    triton_nvls_kernels_allowed = not getattr(
        config, 'inference_disable_triton_nvls_kernels', False
    )

    if triton_nvls_kernels_allowed and SymmetricMemoryManager.is_initialized("tp"):
        buf = SymmetricMemoryManager.get_buffer("tp", process_group=tp_group)
        symm_mem_buffer = buf.maybe_get_tensor(list(x.size()), dtype=x.dtype)

        if (
            x.dtype == torch.bfloat16
            and are_tensors_nvls_eligible(x)
            and symm_mem_buffer["handle"] is not None
        ):
            symm_mem_buffer["tensor"].copy_(x)
            output_dims = list(x.size())
            output_dims[0] = x.size(0) // tp_size
            output = torch.empty(output_dims, dtype=x.dtype, device=x.device)
            multimem_reduce_scatter(output, symm_mem_buffer["tensor"], symm_mem_buffer["handle"])
            return output

    return reduce_scatter_to_sequence_parallel_region(x, group=tp_group)
