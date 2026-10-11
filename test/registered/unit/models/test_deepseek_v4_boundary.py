"""mHC consumer ownership, replicated shared add, and TBO invocation lifetime."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers import dp_attention
from sglang.srt.models import deepseek_v4_boundary as boundary
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common.amd import deepseek_v4_fused_mhc as hip
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHcConsumer(unittest.TestCase):
    def test_dspark_tap_hands_materialized_stream_to_next_layer(self):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4

        class Layer(torch.nn.Module):
            engram = None
            boundary_mhc_fusions = False
            boundary_pre_from_prev = True
            hc_cfg = mhc.HcConfig(2, 4, 1e-5, 1e-6, 4, True, False)

            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, hidden_states, forward_batch, **kwargs):
                stream = residual_batch.stream_of(forward_batch)
                stream.check(hidden_states)
                update = boundary.HcUpdate(self, False)
                update.post = lambda value, state, **kw: mhc.HcState(
                    state.residual + value, torch.ones(3, 2)
                )
                return stream.record(torch.full((3, 2, 4), self.value), update)

        model = deepseek_v4.DeepseekV4Model.__new__(deepseek_v4.DeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.layers = torch.nn.ModuleList([Layer(1.0), Layer(2.0)])
        model.start_layer, model.end_layer = 0, 2
        model.pp_group = SimpleNamespace(world_size=1)
        model.engram_hasher = model.late_layer_start = None
        model.dspark_layers_to_capture = [1]
        positions = torch.arange(3)
        batch = SimpleNamespace(
            positions=positions,
            input_ids=positions,
            forward_mode=ForwardMode.EXTEND,
            residual_stream=None,
        )
        source = torch.arange(24.0).reshape(3, 2, 4)
        aux = []
        recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())
        with (
            patch.object(deepseek_v4, "_is_hip", False),
            patch.object(deepseek_v4, "is_cp_active", return_value=False),
            patch.object(deepseek_v4, "check_cuda_graph_backend", return_value=False),
            patch.object(
                deepseek_v4,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
        ):
            output, pre, tail = model._forward_layers_hc_pre_from_prev(
                positions, source, batch, positions, positions, True, aux
            )
        torch.testing.assert_close(output, source + 3)
        torch.testing.assert_close(aux[0], (source + 1).mean(1))
        torch.testing.assert_close(pre, torch.ones(3, 2))
        self.assertIsNone(tail)
        self.assertIsNone(batch.residual_stream)

    def test_rocm_fused_layer_keeps_legacy_producers_without_stages(self):
        from sglang.srt.models import deepseek_v4

        config = SimpleNamespace(
            hidden_size=8,
            hc_pre_from_prev_sublayer=True,
            model_type="deepseek_v41",
            vision_n_layers=0,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            rms_norm_eps=1e-6,
        )
        attention = SimpleNamespace(wo_b=SimpleNamespace(reduce_results=True))
        moe = SimpleNamespace(reduce_results=True)
        hip_entry = SimpleNamespace(
            init_decoder_layer=lambda layer, quant: setattr(
                layer, "hc_boundary_fused", True
            )
        )
        with (
            patch.object(deepseek_v4, "_is_hip", True),
            patch.object(deepseek_v4, "is_dsa_enable_prefill_cp", return_value=False),
            patch.object(deepseek_v4, "_hip", hip_entry, create=True),
            patch.object(
                deepseek_v4.DeepseekV4DecoderLayer,
                "_build_self_attn",
                return_value=attention,
            ),
            patch.object(deepseek_v4.deepseek_v2, "DeepseekV2MoE", return_value=moe),
            patch.object(
                deepseek_v4,
                "append_stages",
                side_effect=AssertionError("legacy entry declares stages"),
            ),
        ):
            layer = deepseek_v4.DeepseekV4DecoderLayer(config, 0)
        self.assertTrue(layer._legacy_hip_boundary)
        self.assertTrue(layer.self_attn.wo_b.reduce_results)
        self.assertTrue(layer.mlp.reduce_results)
        self.assertFalse(hasattr(layer, "attn_boundary"))
        self.assertTrue(callable(layer._run_moe_ffn_dp_sync))

    def layer(self):
        return SimpleNamespace(
            boundary_mhc_fusions=True,
            boundary_pre_from_prev=True,
            hc_boundary_fused=True,
            hc_attn_fn=object(),
            hc_attn_scale=object(),
            hc_attn_base=object(),
            input_layernorm=object(),
            engram=None,
            _input_norm=Mock(side_effect=lambda x, **kw: (x + 3, None)),
        )

    def test_hip_consumer_owns_pre_and_keeps_producer_coefficients(self):
        layer = self.layer()
        residual, value = torch.randn(2, 4, 8), torch.randn(2, 8)
        pre, post, comb = object(), object(), object()
        previous = boundary.ReadState(
            residual,
            coefficients=SimpleNamespace(tensors=Mock(return_value=(pre, post, comb))),
        )
        coefficients = SimpleNamespace(tensors=Mock())
        batch = object()
        read = boundary.HcRead(layer, True).bind(
            batch, boundary.LayerCall(torch.arange(2))
        )
        with (
            patch.object(
                hip, "hc_boundary", return_value=(residual, value, coefficients)
            ) as kernel,
            patch.object(hip, "attention_mhc_fusion", return_value=None) as fusion,
        ):
            result, state = read.update_and_read(
                boundary.HcUpdate(object(), False), value, previous, None
            )
        kernel.assert_called_once_with(
            layer,
            value,
            residual,
            post,
            comb,
            pre,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
        )
        layer._input_norm.assert_called_once_with(
            value, allow_aiter_quant=False, coefficients=coefficients
        )
        fusion.assert_called_once_with(layer, residual, coefficients, batch)
        torch.testing.assert_close(result, value + 3)
        self.assertIs(state.streams, residual)
        self.assertIs(state.coefficients, coefficients)
        previous.coefficients.tensors.assert_called_once_with()

    def test_hip_final_update_preserves_lagged_pre(self):
        layer = self.layer()
        layer.hc_post = Mock(return_value=torch.ones(2, 4, 8))
        residual, value = torch.randn(2, 4, 8), torch.randn(2, 8)
        pre, post, comb = object(), object(), object()
        state = boundary.ReadState(
            residual, coefficients=SimpleNamespace(tensors=lambda: (pre, post, comb))
        )
        result = boundary.HcUpdate(layer, False).update(value, state)
        layer.hc_post.assert_called_once_with(value, residual, post, comb)
        self.assertIs(result.pre, pre)
        self.assertIs(result.residual, layer.hc_post.return_value)

    def test_initial_hip_read_releases_stale_state(self):
        layer = self.layer()
        residual = torch.randn(2, 4, 8)
        old = mhc.HcState(residual)
        with (
            patch.object(
                hip, "hc_boundary", return_value=(residual, residual[:, 0], object())
            ),
            patch.object(hip, "attention_mhc_fusion", return_value=None),
        ):
            _, current = (
                boundary.HcRead(layer, True).bind(object(), object()).read(old, None)
            )
        self.assertIsNone(old.streams)
        self.assertIs(current.streams, residual)

    def test_replicated_shared_is_added_after_the_sum(self):
        routed = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
        other = torch.tensor([[-1.0, -0.5]], dtype=torch.bfloat16)
        shared = torch.tensor([[0.01, 0.003]], dtype=torch.bfloat16)
        pieces = SimpleNamespace(
            get_merged=lambda: routed, shared=shared, shared_is_replicated=True
        )
        with (
            patch(
                "sglang.srt.layers.moe.post_experts_all_reduce",
                side_effect=lambda x: x + other,
            ) as reduce,
            patch(
                "sglang.srt.layers.moe.utils.should_add_replicated_moe_output",
                return_value=True,
            ),
        ):
            output = boundary.MoeFinalize(pieces).complete()
        reduce.assert_called_once_with(routed)
        torch.testing.assert_close(output, (routed + other) + shared, atol=0, rtol=0)
        self.assertFalse(torch.equal(output, (routed + shared) + other))


class TestOverlappedTransport(unittest.TestCase):
    def test_interleaved_microbatches_keep_events_and_buffers_separate(self):
        trace = []
        compute = SimpleNamespace(
            wait_event=lambda event: trace.append(("wait", event.key))
        )
        comm = SimpleNamespace(wait_stream=lambda stream: trace.append(("fork",)))
        buffers = {}

        def buffer(key, n, h, dtype, device):
            value = torch.empty(n, h, dtype=dtype, device=device)
            buffers[key] = value
            return value

        def gather(out, value, batch):
            trace.append(("gather", batch.sub))
            out.copy_(torch.cat((value, value + 10)))

        def event(key):
            return SimpleNamespace(
                key=key, record=lambda stream: trace.append(("record", key))
            )

        def combine(out, value, sizes, *, event_key):
            trace.append(("combine", event_key))
            out.copy_(value[: out.shape[0]] * 2)
            return event(event_key)

        layer = SimpleNamespace(mlp=SimpleNamespace())
        inputs = (torch.arange(6.0).view(2, 3), torch.tensor([[7.0, 11.0, 13.0]]))
        batches = [SimpleNamespace(sub=i) for i in range(2)]
        lengths = [4, 2]
        with (
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
            patch("sglang.srt.models.deepseek_v4._SHARED_EXPERT_LOCAL", False),
            patch.object(torch.cuda, "current_stream", return_value=compute),
            patch.object(
                torch.cuda, "stream", side_effect=lambda stream: nullcontext()
            ),
            patch.object(dp_attention, "get_dp_tbo_comm_stream", return_value=comm),
            patch.object(dp_attention, "get_tbo_persistent_buffer", side_effect=buffer),
            patch.object(dp_attention, "get_global_dp_buffer_len", side_effect=lengths),
            patch.object(dp_attention, "get_local_dp_buffer_len", side_effect=[2, 1]),
            patch.object(
                dp_attention, "get_dp_global_num_tokens", side_effect=[[2, 2], [1, 1]]
            ),
            patch.object(dp_attention, "dp_gather_partial", side_effect=gather),
            patch.object(dp_attention, "_tbo_event", side_effect=event),
            patch.object(dp_attention, "dp_reduce_scatterv_async", side_effect=combine),
        ):
            calls = [
                boundary.OverlappedMoeTransport(
                    layer,
                    x,
                    batch,
                    boundary.LayerCall(None, tbo=True, tbo_subbatch_index=i),
                )
                for i, (x, batch) in enumerate(zip(inputs, batches))
            ]
            for call in calls:
                with self.assertRaisesRegex(RuntimeError, "combine wait"):
                    call.finish(call.input)
                call.issue_gather()
            for call in calls:
                x = call.wait_gather()
                call.issue_combine(x * 3)
            for call, expected in zip(calls, inputs):
                output = call.wait_combine()
                torch.testing.assert_close(call.finish(output), expected * 6)
                self.assertIsNone(call.gather_keepalive)
                self.assertIsNone(call.combine_keepalive)
                self.assertIsNone(call.global_hidden)
                with self.assertRaisesRegex(RuntimeError, "not issued"):
                    call.wait_combine()
        self.assertEqual(
            [x for x in trace if x[0] == "gather"], [("gather", 0), ("gather", 1)]
        )
        self.assertIsNot(buffers[("gh", 0)], buffers[("gh", 1)])


if __name__ == "__main__":
    unittest.main()
