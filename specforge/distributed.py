"""Process groups for training.

Two axes:

  * The **target** model is tensor-parallel over `tp_size` ranks
    (`get_tp_group()`) and replicated across `world_size / tp_size` data-parallel
    replicas (`get_dp_group()`). The sampler lives on the DP group, so the
    `tp_size` ranks inside one replica see the same batch.
  * The **draft** model has no tensor parallelism. It is wrapped in FSDP over
    the whole world, so every rank holds a shard of its parameters and its
    data-parallel world is every GPU in the job.
"""

from datetime import timedelta

import torch
import torch.distributed as dist

from specforge.utils import print_with_rank

_DEVICE_MESH = None
_TP_DEVICE_MESH = None
_TP_GROUP = None
_DP_GROUP = None


def get_tp_group():
    """Ranks holding shards of one target-model replica."""
    return _TP_GROUP


def get_dp_group():
    """One rank per target-model replica; the dataloader sampler's group."""
    return _DP_GROUP


def get_tp_device_mesh():
    return _TP_DEVICE_MESH


def init_distributed(timeout: int = 10, tp_size: int = 1) -> None:
    """Initialize distributed training.

    Args:
        timeout: Timeout for collective communication, in minutes.
        tp_size: Tensor-parallel degree of the target model.
    """
    global _DEVICE_MESH, _TP_DEVICE_MESH, _TP_GROUP, _DP_GROUP

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    world_size = dist.get_world_size()
    assert world_size % tp_size == 0, (
        f"world size must be divisible by tp size, got world_size={world_size}, "
        f"tp_size={tp_size}"
    )
    dp_size = world_size // tp_size

    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    print_with_rank(f"device mesh: {device_mesh}")

    _DEVICE_MESH = device_mesh
    _TP_GROUP = device_mesh.get_group("tp")
    _DP_GROUP = device_mesh.get_group("dp")
    # A 1D submesh, needed to shard the target model's weights.
    _TP_DEVICE_MESH = dist.DeviceMesh.from_group(_TP_GROUP, device_type="cuda")


def destroy_distributed() -> None:
    dist.destroy_process_group(_TP_GROUP)
    dist.destroy_process_group(_DP_GROUP)
    dist.destroy_process_group()
