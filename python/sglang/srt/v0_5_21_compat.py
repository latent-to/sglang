"""Names SGLang removed after v0.5.21 that submitted DeepSeek-V4.1 arena bundles use.

The arena pin moved from v0.5.21 to the nightly fork, which deleted or renamed
these. Each hosting module resolves a missing name here through a PEP 562
``__getattr__``, so no stock code path changes and a bundle written against
v0.5.21 still imports. Definitions are v0.5.21's, verbatim; a renamed one
aliases the current object. Removed modules are restored as files instead.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable, Optional

import msgspec
import torch

logger = logging.getLogger(__name__)


def removed_attribute(module: str, name: str):
    """Resolve ``module.name`` removed after v0.5.21, else raise the usual AttributeError."""
    loader = _REMOVED.get(module, {}).get(name)
    if loader is None:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    return loader()


def _alias(module: str, name: str) -> Callable[[], object]:
    return lambda: getattr(importlib.import_module(module), name)


# --- v0.5.21 srt/layers/layer_boundary/output.py
class HandoffOutput(msgspec.Struct, frozen=True):
    """A layer output that still owes work only its producer knows how to do (a
    MoE's finalize and sum), left for the next layer's input or for a terminal
    norm that accepts it (residual_batch.norm(handoff_norm=...)). A fused kernel
    there may do that work together with its own; anything else passes it
    through reduce_output(), which calls ``complete()``."""

    def complete(self) -> torch.Tensor:
        """Do the owed work, unfused, and return the complete output."""
        raise NotImplementedError


# --- v0.5.21 srt/layers/layer_boundary/fusions/cutedsl.py
class MoeFinalizeHandoff(HandoffOutput, frozen=True):
    """Unfinalized routed output plus the separately gated shared contribution.
    The next layer's fused finalize + AR + add + norm takes it; ``finish`` is
    the MoE's own unfused tail, for any other reader."""

    routed_output: torch.Tensor
    expert_weights: torch.Tensor
    permuted_indices: torch.Tensor
    gated_shared_output: torch.Tensor
    m: int
    finish: Callable[[], torch.Tensor]

    def complete(self) -> torch.Tensor:
        return self.finish()

    @classmethod
    def from_flashinfer(
        cls,
        deferred_output,
        *,
        gated_shared_output: torch.Tensor,
        m: int,
        reduce: Callable[[torch.Tensor], torch.Tensor],
    ) -> MoeFinalizeHandoff:
        """``reduce`` is the MoE's own sum of its finalized output."""
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            finalize_flashinfer_trtllm_deferred_output,
        )

        top_k = int(deferred_output.top_k)
        return cls(
            routed_output=deferred_output.gemm2_out.view(
                -1, deferred_output.gemm2_out.shape[-1]
            ),
            expert_weights=deferred_output.expert_weights.view(-1, top_k)[:m],
            permuted_indices=deferred_output.expanded_idx_to_permuted_idx.view(
                -1, top_k
            )[:m],
            gated_shared_output=gated_shared_output,
            m=int(m),
            finish=lambda: reduce(
                finalize_flashinfer_trtllm_deferred_output(
                    deferred_output, gated_shared_output
                )
            ),
        )


# --- v0.5.21 srt/layers/attention/dsv4/v41_indexer/types.py
class Selection(msgspec.Struct, frozen=True):
    page_indices: torch.Tensor  # index-K pool slots, ascending
    raw_indices: Optional[torch.Tensor]  # the same picks as compressed positions

    def reset(self) -> None:
        self.page_indices.fill_(-1)
        if self.raw_indices is not None:
            self.raw_indices.fill_(-1)


# --- v0.5.21 srt/layers/layer_boundary/adapters/context_parallel.py
def dsa_cp_gather_hidden_states(hidden_states: torch.Tensor):
    from sglang.srt.layers.dp_attention import (
        attn_cp_all_gather_into_tensor,
        get_local_dp_buffer,
    )
    from sglang.srt.runtime_context import get_parallel

    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_dp_size == 1 and attn_tp_size == 1
    hidden_states, local_hidden_states = (
        get_local_dp_buffer(get_parallel().attn_cp_group),
        hidden_states,
    )
    attn_cp_all_gather_into_tensor(hidden_states, local_hidden_states)
    return hidden_states


def dsa_cp_reduce_scatter_hidden_states(hidden_states: torch.Tensor):
    from sglang.srt.layers.dp_attention import attn_cp_reduce_scatter_tensor
    from sglang.srt.runtime_context import get_parallel

    attn_dp_size = get_parallel().attn_dp_size
    attn_tp_size = get_parallel().attn_tp_size
    assert attn_dp_size == 1 and attn_tp_size == 1
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    input_hidden_states = hidden_states
    hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]
    attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)
    return hidden_states


# --- v0.5.21 kernels/ops/layernorm/mhc.py
def _num_stages_for(m: int, k: int) -> int:
    from sglang.kernels.ops.layernorm.mhc import _HC_MIX_NUM_STAGES
    from sglang.srt.runtime_context import get_platform

    # GB300 verify batches benefit from a smaller shared-memory footprint.
    if get_platform().is_blackwell and k == 20480 and 64 <= m <= 384:
        return 1
    return _HC_MIX_NUM_STAGES


# --- v0.5.21 srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py. Bundles
# read the comm globals through that module at call time, so its __getattr__
# returns their current values from here.
_fused_finalize_all_reduce_world_size: Optional[int] = None
_fused_finalize_all_reduce_probed = False
_fused_finalize_all_reduce_comm = None


def _fused_finalize_all_reduce_comm_world_size() -> Optional[int]:
    """Reserve a separate push plane for up to 384 rows of fused MoE output."""
    global _fused_finalize_all_reduce_world_size, _fused_finalize_all_reduce_probed
    global _fused_finalize_all_reduce_comm
    if not _fused_finalize_all_reduce_probed:
        # The eager warmup must initialize peer workspaces before capture.
        if torch.cuda.is_current_stream_capturing():
            return None
        _fused_finalize_all_reduce_probed = True
        from sglang.kernels.ops.communication import all_reduce_fusion
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils import log_info_on_rank0

        ca_comm = get_parallel().tp_group.ca_comm
        if isinstance(ca_comm, CustomAllReduceV2) and not ca_comm.disabled:
            fused_comm = ca_comm
            if ca_comm.world_size == 4:
                from sglang.kernels.ops.communication.mp import register_comm_cleanup

                fused_comm = CustomAllReduceV2(
                    ca_comm.group,
                    ca_comm.device,
                    max_pull_size=0,
                    max_pull_blocks=0,
                    max_push_size=4 * 1024 * 1024,
                    max_push_blocks=512,
                )
                register_comm_cleanup(fused_comm)
            if fused_comm.disabled:
                return None
            _fused_finalize_all_reduce_comm = fused_comm
            all_reduce_fusion.register_comm(fused_comm.obj)
            _fused_finalize_all_reduce_world_size = fused_comm.world_size
        else:
            log_info_on_rank0(
                logger,
                "Fused MoE finalize: TP group has no "
                "CustomAllReduceV2 push plane; keeping the unfused finalize path",
            )
    return _fused_finalize_all_reduce_world_size


def should_use_fuse_finalize_all_reduce(
    experts, num_tokens: int, hidden_dim: int
) -> bool:
    """Capability only; the batch-size policy cap lives at the call site. The
    kernel never rescales, so the expert weights must carry the routed scaling."""
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        Mxfp4FlashinferTrtllmMoEMethod,
    )
    from sglang.srt.runtime_context import get_parallel

    if not isinstance(experts.quant_method, Mxfp4FlashinferTrtllmMoEMethod):
        return False
    if experts.quant_method.flashinfer_mxfp4_moe_precision != "default":
        return False
    if not experts.should_fuse_routed_scaling_factor_in_topk:
        return False
    if num_tokens <= 0:
        return False
    if num_tokens > 96:
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

        if is_batch_invariant_mode_enabled():
            return False
    from sglang.kernels.ops.communication import all_reduce_fusion

    if not all_reduce_fusion.valid_cluster_sizes(hidden_dim):
        return False
    tp_group = get_parallel().tp_group
    if _fused_finalize_all_reduce_comm_world_size() != tp_group.world_size:
        return False
    comm = _fused_finalize_all_reduce_comm
    # Each token row owns a push phase counter on the epilogue's plane.
    if num_tokens > comm.config.num_push_blocks:
        return False
    return all_reduce_fusion.fits_push_slot(comm.max_push_size, num_tokens, hidden_dim)


_ALL_REDUCE_MHC_V0_5_21 = "sglang.kernels.ops.communication._all_reduce_mhc_v0_5_21"

_REMOVED = {
    "sglang.kernels.ops.attention.dsv4.wo_a": {
        "GROUPS": lambda: 2,
        "N_OUT": lambda: 2048,  # GROUPS * RANK
        "SCALE_BYTES": lambda: 8192,
    },
    "sglang.kernels.ops.communication.all_reduce_mhc": {
        "all_reduce_mhc_norm": _alias(_ALL_REDUCE_MHC_V0_5_21, "all_reduce_mhc_norm"),
        "moe_finalize_all_reduce_mhc": _alias(_ALL_REDUCE_MHC_V0_5_21, "moe_finalize_all_reduce_mhc"),
        "moe_finalize_all_reduce_mhc_norm": _alias(_ALL_REDUCE_MHC_V0_5_21, "moe_finalize_all_reduce_mhc_norm"),
        "moe_finalize_all_reduce_mhc_quant": _alias(_ALL_REDUCE_MHC_V0_5_21, "moe_finalize_all_reduce_mhc_quant"),
    },
    "sglang.kernels.ops.layernorm.mhc": {
        "_num_stages_for": lambda: _num_stages_for,
    },
    "sglang.srt.layers.attention.dsv4.v41_indexer.sparse_table": {
        "Selection": lambda: Selection,
    },
    "sglang.srt.layers.layer_boundary.fusions.cutedsl": {
        "MoeFinalizeHandoff": lambda: MoeFinalizeHandoff,
    },
    "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe": {
        "_fused_finalize_all_reduce_comm": lambda: _fused_finalize_all_reduce_comm,
        "_fused_finalize_all_reduce_comm_world_size": lambda: _fused_finalize_all_reduce_comm_world_size,
        "_fused_finalize_all_reduce_probed": lambda: _fused_finalize_all_reduce_probed,
        "_fused_finalize_all_reduce_world_size": lambda: _fused_finalize_all_reduce_world_size,
        "should_use_fuse_finalize_all_reduce": lambda: should_use_fuse_finalize_all_reduce,
    },
    "sglang.srt.models.deepseek_v2": {
        "_FUSED_FINALIZE_ALL_REDUCE_MAX_TOKENS": lambda: 384,
        "should_skip_post_experts_all_reduce": _alias("sglang.srt.layers.moe.utils", "should_skip_post_experts_all_reduce"),
        "should_use_fuse_finalize_all_reduce": lambda: should_use_fuse_finalize_all_reduce,
    },
    "sglang.srt.models.deepseek_v4": {
        "_get_mhc_ops": _alias("sglang.srt.models.deepseek_v4", "get_mhc_ops"),
        "dsa_cp_gather_hidden_states": lambda: dsa_cp_gather_hidden_states,
        "dsa_cp_reduce_scatter_hidden_states": lambda: dsa_cp_reduce_scatter_hidden_states,
        "dsa_use_prefill_cp": _alias("sglang.srt.layers.attention.dsa.utils", "dsa_use_prefill_cp"),
    },
}
