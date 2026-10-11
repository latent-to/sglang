"""A custom output sum, channelwise convolution, and terminal read share one owner."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.layer_boundary import (
    ReadoutFusion,
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
    prepare,
)
from sglang.srt.layers.layer_boundary.facts import facts_of
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.output import SumImplementation, UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layer_boundary.residual import bind_update
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.short_conv import (
    ShortConvRead,
    ShortConvUpdate,
)
from sglang.srt.layers.layer_boundary.residual.stream import (
    OwedOutput,
    ResidualStream,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.communicator_patch import patch_communicator

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestShortConvBoundary(unittest.TestCase):
    def test_sum_conv_gather_add_norm_and_tail(self):
        for defer in (False, True):
            for scattered in (False, True):
                with self.subTest(defer=defer, scattered=scattered):
                    self.run_chain(defer, scattered)

    def test_custom_completion_keeps_its_lora_deferral_gate(self):
        self.run_chain(True, False, lora=True)

    def run_chain(self, defer, scattered, lora=False):
        calls = []
        batch = SimpleNamespace(
            positions=torch.tensor([3, 8]),
            input_ids=torch.tensor([1, 2]),
            forward_mode=ForwardMode.DECODE,
            residual_stream=None,
        )

        def complete(value, forward_batch):
            calls.append("sum")
            value = value * 2 + 0.7
            return value[:, :2] if scattered else value

        def conv(value, positions, forward_batch):
            self.assertIs(forward_batch, batch)
            calls.append("conv")
            return value.square() * 0.1 + positions[:, None]

        def gather(value):
            calls.append("hidden_gather")
            return torch.cat((value, value + 5), dim=-1)

        norm = fixture.Norm()
        read = ShortConvRead()
        update = ShortConvUpdate(conv, None, SumGroup.TP, gather if scattered else None)
        custom = SumImplementation(complete, lambda fb: defer)
        parallel = fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=1)
        with (
            fixture.planning(parallel, boundary_reduction="ar"),
            patch_communicator("get_lora", lambda: SimpleNamespace(enable_lora=lora)),
            layer_stack(final_read=read),
        ):
            stages = [
                stage
                for _ in range(2)
                for stage in append_stages(
                    (declare_attn(read=read, update=update, sum_impl=custom), norm),
                    (declare_ffn(read=read, update=update, sum_impl=custom), norm),
                )
            ]
        initial = torch.tensor([[0.2, -0.4, 0.6, 0.8], [-0.7, 0.9, 0.1, 0.3]])
        hidden, residual = initial, initial
        residual_batch.start(batch)
        with (
            fixture.planning(parallel, boundary_reduction="ar"),
            patch_communicator("get_lora", lambda: SimpleNamespace(enable_lora=lora)),
        ):
            for index, stage in enumerate(stages):
                hidden = stage.prepare(hidden, batch)
                torch.testing.assert_close(hidden, 2 * residual)
                output = hidden * (0.2 + 0.03 * index)
                hidden = stage.finish(output, batch)
                expected = output * 2 + 0.7
                if scattered:
                    expected = expected[:, :2]
                expected = expected.square() * 0.1 + batch.positions[:, None]
                if scattered:
                    expected = torch.cat((expected, expected + 5), dim=-1)
                residual = residual + expected
            if defer:
                self.assertIsInstance(hidden, OwedOutput)
            result, tail = residual_batch.read_output(hidden, batch, read, norm)
        torch.testing.assert_close(result, 2 * residual)
        torch.testing.assert_close(tail, residual)
        self.assertEqual(calls.count("sum"), 4)
        self.assertEqual(calls.count("conv"), 4)
        self.assertEqual(calls.count("hidden_gather"), 4 if scattered else 0)
        self.assertIsNone(batch.residual_stream)
        if isinstance(hidden, OwedOutput):
            self.assertIsNone(hidden.contribution.value)
            self.assertIsNone(hidden.contribution.owed)

    def test_fusion_receives_the_actual_update_and_rejects_before_mutation(self):
        update = ShortConvUpdate(
            lambda *args: self.fail("unfused conv"), None, SumGroup.TP
        )
        actual = bind_update(update, SimpleNamespace())
        events = []
        expected = (torch.ones(2, 4), torch.zeros(2, 4))

        def run(*args, update):
            events.append(update)
            return expected

        rejected = ReadoutFusion(
            SumGroup.TP,
            lambda *args, **kwargs: self.fail("rejected fusion ran"),
            accepts_update=lambda u: False,
        )
        accepted = ReadoutFusion(
            SumGroup.TP, run, reads=True, accepts_update=lambda u: u is actual
        )
        result = prepare._run_read_fusions(
            (rejected, accepted),
            torch.ones(2, 4),
            torch.zeros(2, 4),
            None,
            None,
            read=ShortConvRead(),
            update=actual,
        )
        self.assertIs(result, expected)
        self.assertEqual(events, [actual])

    def test_plain_callback_signature_is_unchanged(self):
        expected = (torch.ones(1, 4), torch.zeros(1, 4))
        fusion = ReadoutFusion(SumGroup.TP, lambda h, r, fb, norm: expected, reads=True)
        self.assertIs(
            prepare._run_read_fusions(
                (fusion,), None, None, None, None, read=None, update=PLAIN_ADD
            ),
            expected,
        )

    def test_carried_group_mismatch_uses_its_own_completion(self):
        group = SimpleNamespace()
        actual = bind_update(
            ShortConvUpdate(lambda h, p, fb: h, group, SumGroup.TP),
            SimpleNamespace(positions=torch.arange(2)),
        )
        owed = UnreducedOutput(torch.ones(2, 4), group, complete_sum=lambda h: h * 3)
        fused = ReadoutFusion(
            SumGroup.ATTN_TP,
            lambda *args, **kw: self.fail("wrong group"),
            reads=True,
            accepts_update=lambda u: True,
            carried=True,
        )
        with patch.object(prepare, "_sum_group", lambda group: object()):
            output, residual = prepare._run_entry(
                owed,
                torch.ones(2, 4),
                None,
                fixture.Norm(),
                step=lambda h, r, fb, norm, update: norm(h, r),
                is_plain_add=False,
                carried_fusions=(),
                carried_read_fusions=(fused,),
                read=ShortConvRead(),
                update=actual,
            )
        torch.testing.assert_close(output, torch.full((2, 4), 8.0))
        torch.testing.assert_close(residual, torch.full((2, 4), 4.0))

    def test_call_context_is_not_reused_between_invocations(self):
        update = ShortConvUpdate(lambda h, p, fb: h + p[:, None], None, SumGroup.TP)
        a, b = (
            SimpleNamespace(positions=torch.tensor([2, 4])),
            SimpleNamespace(positions=torch.tensor([7, 11])),
        )
        first, second = bind_update(update, a), bind_update(update, b)
        value = torch.zeros(2, 4)
        torch.testing.assert_close(
            first.contribution(value), a.positions[:, None].expand_as(value).to(value)
        )
        torch.testing.assert_close(
            second.contribution(value), b.positions[:, None].expand_as(value).to(value)
        )
        torch.testing.assert_close(
            first.contribution(value), a.positions[:, None].expand_as(value).to(value)
        )

    def test_consumed_handle_does_not_retain_the_forward_batch(self):
        class Batch:
            positions = torch.arange(2)

        batch = Batch()
        batch.residual_stream = ResidualStream(torch.ones(2, 4))
        update = bind_update(
            ShortConvUpdate(lambda h, p, fb: h.square(), None, SumGroup.TP), batch
        )
        handle = batch.residual_stream.record(
            UnreducedOutput(torch.ones(2, 4), complete_sum=lambda h: h * 2), update
        )
        reference = weakref.ref(batch)
        residual_batch.read_output(handle, batch, ShortConvRead(), fixture.Norm())
        del batch, update
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(handle.contribution.update)

    def test_explicit_positions_belong_to_the_producer_invocation(self):
        batch = SimpleNamespace(positions=torch.tensor([2, 4]))
        positions = torch.tensor([17, 23])
        update = bind_update(
            ShortConvUpdate(lambda h, p, fb: h + p[:, None], None, SumGroup.TP),
            batch,
            positions,
        )
        value = torch.zeros(2, 4)
        batch.positions = torch.tensor([101, 103])
        torch.testing.assert_close(
            update.contribution(value), positions[:, None].expand_as(value).to(value)
        )

    def test_final_read_requires_explicit_acceptance_and_releases_on_failure(self):
        update = bind_update(
            ShortConvUpdate(lambda *args: self.fail("conv"), None, SumGroup.TP),
            SimpleNamespace(),
        )
        batch = SimpleNamespace(residual_stream=ResidualStream(torch.ones(2, 4)))
        output = batch.residual_stream.record(
            torch.zeros(2, 4), update, declared_sum=SumGroup.TP
        )
        with self.assertRaises(NotImplementedError):
            residual_batch.read_output(output, batch, SimpleNamespace(), None)
        self.assertIsNone(batch.residual_stream)
        self.assertIsNone(output.contribution.value)

    def test_terminal_nonplain_update_still_rejected_without_final_read(self):
        update = ShortConvUpdate(lambda *args: None, None, SumGroup.TP)
        with fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=1)):
            with self.assertRaises(NotImplementedError), layer_stack():
                append_stages(
                    (declare_ffn(read=ShortConvRead(), update=update), fixture.Norm())
                )

    def test_neighbour_facts_do_not_keep_the_sum_implementation(self):
        declaration = declare_ffn(
            sum_impl=SumImplementation(lambda *args: None, lambda fb: True)
        )
        facts = facts_of(declaration)
        self.assertIsNone(facts.sum_impl.run)
        self.assertIsNone(facts.sum_impl.can_defer)


if __name__ == "__main__":
    unittest.main()
