"""Transport a compute operand without moving its source residual."""

from contextlib import nullcontext
from typing import Callable, Mapping, Optional

import msgspec

from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup
from sglang.srt.layers.layer_boundary.output import DeferredFinalize, UnreducedOutput
from sglang.srt.layers.layer_boundary.residual import bind_update


class TransportRows(msgspec.Struct, frozen=True):
    """The source/read, compute, and completed output rows of one adapter.

    The residual stays on source rows. ``group`` names the compute output's
    outstanding sum; None means the computation (e.g. EP combine) completes it.
    The adapter finishes that sum and returns source rows, or publishes an
    explicit completion handle on source rows. It never applies the update.
    """

    source: Layout
    compute: Layout
    returned: Layout
    group: Optional[SumGroup] = None


class RowTransport(msgspec.Struct, frozen=True):
    """Declare separate compute/residual rows and bind one invocation's moves.

    ``bind(input, batch, context)`` returns an invocation with ``input``,
    ``scope()`` and ``finish(output)``. Its scope publishes compute's reduction
    policy; finish owns the sum and inverse movement. Facts retain rows only.
    """

    rows: Mapping[BatchVariant, TransportRows]
    bind: Optional[Callable] = None
    overlapped_rows: Optional[Mapping[BatchVariant, TransportRows]] = None

    def facts(self):
        return RowTransport(
            dict(self.rows),
            overlapped_rows=(
                dict(self.overlapped_rows) if self.overlapped_rows is not None else None
            ),
        )


class ComputeTransport:
    """One prepared compute operand; finish owns its sum and return movement.

    Subclasses publish ``input`` and implement finish(output). A result must
    be complete on the declared return rows, or an explicit completion handle.
    Residual update and consumer normalization remain outside this adapter.
    """

    @staticmethod
    def scope():
        return nullcontext()

    def finish(self, output):
        raise NotImplementedError


class TransportExit:
    """Consume a prepared transport once, then publish the residual update."""

    def __init__(self, stage, batch):
        self.stage = stage
        self.batch = batch
        self.stream = batch.residual_stream
        owner, self.invocation = self.stream.transport
        if owner is not stage:
            raise RuntimeError("compute transport belongs to another stage")
        self.scope = self.invocation.scope()
        self.used = False

    def __enter__(self):
        try:
            self.scope.__enter__()
        except BaseException:
            self.stream.transport = None
            raise
        return self

    def __exit__(self, *exc):
        try:
            result = self.scope.__exit__(*exc)
        finally:
            self.stream.transport = None
        if exc[0] is None and not self.used:
            raise RuntimeError("compute transport exited without finish")
        return result

    def finish(self, output):
        if self.used:
            raise RuntimeError("compute transport was already consumed")
        self.used = True
        output = self.invocation.finish(output)
        produced = self.stage.plan.path_for(self.batch).output
        update = bind_update(produced.update, self.batch)
        if (
            update.applied_at_exit
            or self.stage.plan.path_for(self.batch).writes_at_handoff
        ):
            if isinstance(output, (DeferredFinalize, UnreducedOutput)):
                output = output.complete()
            return self.stream.write(update.update(output, self.stream.residual))
        return self.stream.record(output, update)
