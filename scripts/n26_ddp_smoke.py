#!/usr/bin/env python3
"""One-batch NCCL/DDP regression smoke for N26; not a training result."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n26_ccsam_model import CCSAM, CCSAMConfig
from n26_train_ccsam import DenseDataset, compute_loss, to_device


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dataset = DenseDataset(Path("outputs/n26/dense_dataset/round0_train30.npz"))
    indices = list(range(rank * 8, rank * 8 + 8))
    batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=8)))
    batch = to_device(batch, device)
    model = DDP(CCSAM(CCSAMConfig()).to(device), device_ids=[local_rank], broadcast_buffers=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _, _ = compute_loss(model, batch)
    loss.backward()
    finite = torch.tensor(float(torch.isfinite(loss) and all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())), device=device)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"N26_DDP_SMOKE_OK world_size={dist.get_world_size()} finite={int(finite.item())}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
