"""
Time data loading vs. model compute for a DHS training config.

Builds the datamodule and model the same way train_dhs.py does, then measures:
  1. per-item dataset time in the main process (and how much of it is footprint tabix reads,
     for profile datasets)
  2. DataLoader throughput with N workers (data only, no model)
  3. GPU time for forward + backward + optimizer step on batches already on the GPU
  4. the real interleaved loop: how long each step blocks waiting for the next batch

If (2) s/batch >> (3) s/batch, training is data-bound; (4) shows the resulting GPU idle time.
"""
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch

from vinson.from_config import datamodule_from_config, read_configs, dhs_model_from_config
from vinson.run import set_global_seed, set_worker_seed


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def time_items(dataset, n_items, seed=0):
    idx = np.random.default_rng(seed).choice(len(dataset), size=n_items, replace=False)
    fp_time = 0.0
    # profile datasets read obs/exp footprints from tabix per item; time that part separately
    extract = getattr(dataset, 'extract_obs_exp_data', None)
    if extract is not None:
        def timed_extract(*args, **kwargs):
            nonlocal fp_time
            t = time.perf_counter()
            out = extract(*args, **kwargs)
            fp_time += time.perf_counter() - t
            return out
        dataset.extract_obs_exp_data = timed_extract

    t = time.perf_counter()
    for i in idx:
        dataset[int(i)]
    total = time.perf_counter() - t
    return total / n_items, (fp_time / n_items if extract is not None else None)


def time_loader(loader, n_batches, n_warmup):
    it = iter(loader)
    for _ in range(n_warmup):
        next(it)
    t = time.perf_counter()
    for _ in range(n_batches):
        next(it)
    return (time.perf_counter() - t) / n_batches


def time_model(model, batches, n_warmup, device):
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def train_step(batch):
        optimizer.zero_grad(set_to_none=True)
        loss = model.step(batch)[0]
        loss.backward()
        optimizer.step()

    for b in batches[:n_warmup]:
        train_step(b)
    torch.cuda.synchronize(device)
    t = time.perf_counter()
    for b in batches[n_warmup:]:
        train_step(b)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t) / (len(batches) - n_warmup)


def time_training_loop(model, loader, n_batches, n_warmup, device):
    """
    Interleave loading and training like the real loop: time blocked in next(loader)
    is how long the GPU sits idle waiting for data.
    """
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    it = iter(loader)
    wait, step = [], []
    for i in range(n_warmup + n_batches):
        t0 = time.perf_counter()
        batch = to_device(next(it), device)
        t1 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model.step(batch)[0]
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        if i >= n_warmup:
            wait.append(t1 - t0)
            step.append(t2 - t1)
    return np.array(wait), np.array(step)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("anndata_file", type=str)
    parser.add_argument("fasta_file", type=str)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--genotype_file", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=11)
    parser.add_argument("--n_items", type=int, default=500, help="Items to time in the main process")
    parser.add_argument("--n_batches", type=int, default=50, help="Batches to time for loader and model")
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    default_config_path = os.path.dirname(os.path.abspath(__file__)) + "/default_train_dhs.config.yaml"
    config = read_configs(default_config_path, overwrite_config_path=args.config)

    datamodule = datamodule_from_config(
        config,
        anndata_file=args.anndata_file,
        fasta_file=args.fasta_file,
        genotype_file=args.genotype_file,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=set_worker_seed,
        persistent_workers=False,
    )
    print("Setting up datamodule...", flush=True)
    t = time.perf_counter()
    datamodule.setup("fit")
    print(f"setup: {time.perf_counter() - t:.1f} s", flush=True)

    batch_size = datamodule.dataloader_kwargs['batch_size']
    loader = datamodule.train_dataloader()

    per_item, per_item_fp = time_items(loader.dataset, args.n_items, seed=args.seed)
    print(f"\n[1] dataset item (main process): {per_item * 1e3:.1f} ms/item "
          f"-> {per_item * batch_size:.2f} s/batch with 1 worker", flush=True)
    if per_item_fp is not None:
        print(f"    of which footprint tabix reads: {per_item_fp * 1e3:.1f} ms/item "
              f"({per_item_fp / per_item:.0%})", flush=True)

    loader_time = time_loader(loader, args.n_batches, args.n_warmup)
    print(f"[2] DataLoader, {args.num_workers} workers: {loader_time:.3f} s/batch", flush=True)

    device = torch.device("cuda")
    model = dhs_model_from_config(config)
    it = iter(datamodule.train_dataloader())
    batches = [to_device(next(it), device) for _ in range(args.n_batches + args.n_warmup)]
    model_time = time_model(model, batches, args.n_warmup, device)
    print(f"[3] model fwd+bwd+step on GPU: {model_time:.3f} s/batch", flush=True)
    del batches

    wait, step = time_training_loop(model, datamodule.train_dataloader(), args.n_batches, args.n_warmup, device)
    print(f"[4] real loop: waiting for data {wait.mean():.3f} s/batch (median {np.median(wait):.3f}, "
          f"max {wait.max():.3f}), training {step.mean():.3f} s/batch "
          f"-> GPU idle {wait.sum() / (wait.sum() + step.sum()):.0%} of the time", flush=True)

    n_batches_epoch = len(loader)
    bound = "DATA" if loader_time > model_time else "GPU"
    print(f"\nbatch_size={batch_size}, {n_batches_epoch} batches/epoch")
    print(f"epoch estimate: data-only {loader_time * n_batches_epoch / 3600:.1f} h, "
          f"model-only {model_time * n_batches_epoch / 3600:.1f} h -> {bound}-bound")
