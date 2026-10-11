"""S1 E5: load-aware routed-expert pruning at decode / verify (a PRECISION CHANGE), DeepSeek-V4.1-Flash.

The GLM-5.3 arena's accepted lever (route_topk v119 / v120 in the GLM crown line) ported to V4.1's routing. In one MoE
call (all of a decode / verify batch's rows; routing is replicated on both TP2 / EP2 ranks, so both decide alike):
  1. an expert picked by <= PRUNE_MAX_COUNT rows is eligible;
  2. each row drops its eligible picks lightest first (ties: the lower slot) while its dropped weight stays within
     PRUNE_BUDGET of the row's routed mass (the weights after stock's renormalization);
  3. an expert is dropped only when EVERY row that picked it dropped it (so a drop always removes the expert's whole
     weight slice from the call; a lone row's drop of an expert others keep would cost accuracy and save nothing).
A dropped pick becomes id -1 / weight 0 -- what stock's own padded rows carry (mask_topk_ids_padded_region), which the
FlashInfer routed MoE skips on both EP ranks and the fused finalize skips (rows < 0). The kept weights are not
renormalized. One Triton launch per MoE layer, after stock's router; the router itself is stock's.
Applied by the bundle's copies of DeepseekV2MoE.forward_normal (eager, the audit's path) and forward_normal_dual_stream
(graph capture) alike, so the node audit grades exactly what is served. Hash-routed layers are left alone."""
import torch
import triton
import triton.language as tl

PRUNE_BUDGET = 0.4        # share of a row's routed mass it may drop
PRUNE_MAX_COUNT = 1        # an expert picked by more rows than this is never dropped
PRUNE_MAX_ROWS = 256       # decode / verify batches only (max_running 24 x 6 verify rows = 144)
_BR = 256                  # most rows a launch covers (one program; the tile is the batch's power of two, >= 8)
_KP = 8                    # top-k slots, padded to a power of two
_NB = 512                  # histogram bins: expert id + 1 (bin 0 = no pick), 384 experts
_on: list = []             # [True] while run_moe has a decode / verify batch in flight
_scratch: dict = {}


@triton.jit
def _prune_kernel(ids_ptr, w_ptr, packed_ptr, R, s_ids, s_w, s_packed, hist_ptr, cand_ptr, budget, max_count,
                  K: tl.constexpr, KP: tl.constexpr, BR: tl.constexpr, NB: tl.constexpr, PACKED: tl.constexpr):
    r = tl.arange(0, BR)[:, None]
    k = tl.arange(0, KP)[None, :]
    m = (r < R) & (k < K)
    ids = tl.load(ids_ptr + r * s_ids + k, mask=m, other=-1)
    w = tl.load(w_ptr + r * s_w + k, mask=m, other=0.0).to(tl.float32)
    valid = ids >= 0
    bins = tl.where(valid, ids + 1, 0)
    # 1. how many rows picked each expert (a row picks an expert at most once)
    tl.store(hist_ptr + tl.arange(0, NB), tl.histogram(tl.reshape(bins, [BR * KP]), NB))
    tl.debug_barrier()
    cnt = tl.load(hist_ptr + bins)
    elig = valid & (cnt <= max_count)
    # 2. a row's eligible picks in (weight, slot) order; the running dropped weight through each pick
    wi = tl.reshape(w, [BR, KP, 1])
    wj = tl.reshape(w, [BR, 1, KP])
    ki = tl.arange(0, KP)[None, :, None]
    kj = tl.arange(0, KP)[None, None, :]
    before = tl.reshape(elig, [BR, KP, 1]) & ((wi < wj) | ((wi == wj) & (ki < kj)))
    run = tl.sum(tl.where(before, wi, 0.0), axis=1) + w
    mass = tl.sum(tl.where(valid, w, 0.0), axis=1)[:, None]
    cand = elig & (run <= budget * mass)
    # 3. an expert goes only if every row that picked it chose to drop it
    tl.store(cand_ptr + tl.arange(0, NB), tl.histogram(tl.reshape(tl.where(cand, bins, 0), [BR * KP]), NB))
    tl.debug_barrier()
    drop = cand & (tl.load(cand_ptr + bins) == cnt)
    tl.store(ids_ptr + r * s_ids + k, tl.where(drop, -1, ids), mask=m)
    tl.store(w_ptr + r * s_w + k, tl.where(drop, 0.0, w), mask=m)
    if PACKED:   # FlashInfer's packed (id << 16 | bf16 weight bits): id -1, weight 0
        p = tl.load(packed_ptr + r * s_packed + k, mask=m, other=0)
        tl.store(packed_ptr + r * s_packed + k, tl.where(drop, -65536, p), mask=m)


def prune_ids(ids: torch.Tensor, weights: torch.Tensor, packed=None) -> None:
    """In place: E5 on one MoE call's routing (ids int32 [R, K], weights fp32 [R, K], optional packed int32 [R, K])."""
    rows, k = ids.shape
    if not (ids.dtype == torch.int32 and weights.dtype == torch.float32 and weights.shape == ids.shape
            and ids.stride(1) == 1 and weights.stride(1) == 1 and k <= _KP and 0 < rows <= _BR):   # noqa: E501
        raise RuntimeError(f"E5 prune: routing {ids.dtype} {tuple(ids.shape)} / {weights.dtype} {tuple(weights.shape)} "
                           "is outside the kernel's domain")
    if packed is not None and not (packed.dtype == torch.int32 and packed.shape == ids.shape and packed.stride(1) == 1):
        raise RuntimeError(f"E5 prune: packed routing {packed.dtype} {tuple(packed.shape)} does not match the ids")
    scratch = _scratch.get(ids.device)
    if scratch is None:
        scratch = _scratch[ids.device] = torch.empty(2, _NB, dtype=torch.int32, device=ids.device)
    _prune_kernel[(1,)](ids, weights, packed if packed is not None else ids, rows, ids.stride(0), weights.stride(0),
                        packed.stride(0) if packed is not None else 0, scratch[0], scratch[1], float(PRUNE_BUDGET),
                        int(PRUNE_MAX_COUNT), K=k, KP=_KP, BR=max(8, triton.next_power_of_2(rows)), NB=_NB,
                        PACKED=packed is not None, num_warps=4 if rows <= 32 else 8)


def prune_topk(moe, topk_output):
    """The bundle's MoE copies call this right after stock's router (S1 E5)."""
    if not _on or moe.is_hash or moe.is_nextn or PRUNE_BUDGET <= 0.0 or PRUNE_MAX_COUNT <= 0:
        return topk_output
    ids = topk_output.topk_ids
    if ids.shape[0] == 0 or ids.shape[0] > PRUNE_MAX_ROWS:
        return topk_output
    prune_ids(ids, topk_output.topk_weights, getattr(topk_output, "packed_topk_ids", None))
    return topk_output


class decode_rows:
    """run_moe: E5 is on for the MoE call of a decode / verify batch, off otherwise (prefill is never pruned)."""

    def __init__(self, forward_batch):
        mode = getattr(forward_batch, "forward_mode", None)
        self.on = mode is not None and (mode.is_decode() or mode.is_target_verify())

    def __enter__(self):
        if self.on:
            _on.append(True)

    def __exit__(self, *exc):
        if self.on:
            _on.pop()
