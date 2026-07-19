import os
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def _env_int(name, default=None):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _first_slurm_host():
    nodelist = os.environ.get("SLURM_NODELIST")
    if not nodelist:
        return None
    try:
        output = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    hosts = [line.strip() for line in output.splitlines() if line.strip()]
    return hosts[0] if hosts else None


def add_distributed_args(parser):
    parser.add_argument("--distributed", default="auto", choices=["auto", "none", "ddp"])
    parser.add_argument("--ddp_backend", "--ddp-backend", dest="ddp_backend", default="auto", choices=["auto", "nccl", "gloo"])
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=None)
    parser.add_argument("--ddp_timeout_min", "--ddp-timeout-min", dest="ddp_timeout_min", type=int, default=60)


def setup_distributed(args):
    world_size = _env_int("WORLD_SIZE", _env_int("SLURM_NTASKS", 1))
    requested = getattr(args, "distributed", "auto")
    enabled = requested == "ddp" or (requested == "auto" and world_size > 1)

    if not enabled:
        return {
            "enabled": False,
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "is_main": True,
            "backend": None,
        }

    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ.setdefault("RANK", "0")
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ.setdefault("LOCAL_RANK", "0")
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = _first_slurm_host() or "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29500")

    local_rank = getattr(args, "local_rank", None)
    if local_rank is None:
        local_rank = _env_int("LOCAL_RANK", _env_int("SLURM_LOCALID", 0))
    local_rank = int(local_rank or 0)

    backend = getattr(args, "ddp_backend", "auto")
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    device_index = None
    if backend == "nccl" and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_index = local_rank if local_rank < device_count else 0
        torch.cuda.set_device(device_index)

    timeout = timedelta(minutes=max(int(getattr(args, "ddp_timeout_min", 60)), 1))
    dist.init_process_group(backend=backend, init_method="env://", timeout=timeout)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return {
        "enabled": True,
        "rank": rank,
        "world_size": world_size,
        "local_rank": int(local_rank or 0),
        "device_index": device_index,
        "is_main": rank == 0,
        "backend": backend,
    }


def resolve_device(requested, dist_info):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")

    if requested == "cuda" and dist_info.get("enabled", False):
        local_rank = int(dist_info.get("local_rank", 0))
        device_count = torch.cuda.device_count()
        cuda_index = dist_info.get("device_index")
        if cuda_index is None:
            cuda_index = local_rank if local_rank < device_count else 0
        torch.cuda.set_device(cuda_index)
        return torch.device("cuda", cuda_index)
    return torch.device(requested)


def wrap_model(model, dist_info, device):
    if not dist_info.get("enabled", False):
        return model
    if device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
        )
    return DistributedDataParallel(model, broadcast_buffers=False)


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def rank_zero_print(dist_info, *args, **kwargs):
    if dist_info.get("is_main", True):
        print(*args, **kwargs)


def distributed_mean_scalars(values, device, dist_info):
    if not dist_info.get("enabled", False):
        return values
    if not values:
        return values
    keys = sorted(values)
    tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(dist_info["world_size"])
    return {key: float(value) for key, value in zip(keys, tensor.detach().cpu().tolist())}


def barrier(dist_info):
    if dist_info.get("enabled", False):
        if dist_info.get("backend") == "nccl" and torch.cuda.is_available():
            device_index = dist_info.get("device_index")
            if device_index is None:
                device_index = int(dist_info.get("local_rank", 0))
            dist.barrier(device_ids=[int(device_index)])
        else:
            dist.barrier()


def cleanup_distributed(dist_info):
    if dist_info.get("enabled", False) and dist.is_initialized():
        dist.destroy_process_group()
