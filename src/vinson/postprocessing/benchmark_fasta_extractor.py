"""
Compares matmul precision settings (fp32 baseline / TF32 / bf16 autocast) for DHS model
inference, using the same dataloader/model path as nextflow/bin/dhs/predict_DHS_model.py.
Earlier profiling on this script (see git history) showed data loading (fasta_disk vs
InMemoryFastaExtractor) is not the bottleneck -- compute_sec is ~99% of steady_state_sec -- so
the thing being compared across modes here is GPU compute precision instead. Fasta backend is
fixed for the whole run (disk by default; pass --fasta_in_memory to use InMemoryFastaExtractor
instead), not itself varied across modes. Reports warmup time, steady-state throughput,
data_wait vs compute split, and CPU/RAM (GB used)/GPU (GB) for each mode, and checks predictions
stay numerically close to the fp32 baseline.

Regions/sample are taken directly from the training AnnData (no h5 export needed): every DHS
region (`adata.var`) is scored against the first sample (`adata.obs_names[0]`).
"""
import argparse
import contextlib
import csv
import shutil
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vinson.from_config import read_configs, dhs_model_from_config
from vinson.utils.data_formatting import VinsonData
from vinson.datasets.sequence import InferenceSequenceEmbedDataset

from genome_tools.data.anndata import read_zarr_backed

GB = 1024 ** 3


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# (mode_name, torch.set_float32_matmul_precision arg, autocast dtype or None)
MODES = [
    ("fp32_baseline", "highest", None),
    ("tf32", "high", None),
    ("bf16_autocast", "high", torch.bfloat16),
]


class UtilizationMonitor:
    """Samples RAM/GPU memory (GB) in a background thread for the duration of a `with` block."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.ram_used_gb = []
        self.gpu_mem_gb = []
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _poll_gpu(self):
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        for mem_mb in out.strip().splitlines():
            self.gpu_mem_gb.append(float(mem_mb) / 1024)

    def _run(self):
        while not self._stop.is_set():
            self.ram_used_gb.append(psutil.virtual_memory().used / GB)
            if self._has_nvidia_smi:
                try:
                    self._poll_gpu()
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @staticmethod
    def _agg(values):
        return (max(values), sum(values) / len(values)) if values else (float("nan"), float("nan"))

    def summary(self):
        ram_max, ram_mean = self._agg(self.ram_used_gb)
        gpu_mem_max, gpu_mem_mean = self._agg(self.gpu_mem_gb)
        return {
            "ram_used_gb_mean": ram_mean, "ram_used_gb_max": ram_max,
            "gpu_mem_gb_mean": gpu_mem_mean, "gpu_mem_gb_max": gpu_mem_max,
        }


def subset_data(data: VinsonData, n: int) -> VinsonData:
    if n is None or n >= len(data):
        return data
    sliced = {key: value[:n] for key, value in data.data.items()}
    return VinsonData(sliced, data.encodings, data.embeddings_df, is_inference=data.is_inference)


def run_mode(mode_name, matmul_precision, autocast_dtype, vinson_data, model, args):
    """
    Excludes the one-time fasta load from steady-state throughput (see module docstring for why
    fasta backend is no longer the variable here). DataLoader assigns batches to workers
    round-robin, so with num_workers=N, each of the first N batches blocks on a *different*
    worker's one-time dataset init -- timing just the first batch (as if num_workers=1) would
    still leave most of that cost smeared across "steady state". So the first `num_workers`
    batches (at least 1) are treated as warmup and timed separately. A standalone priming pass
    isn't usable here since it would spin up its own (redundant) workers, so the scoring loop
    from vinson.inference.score_bed_with_dhs_model is inlined to split timing within a single
    iterator.
    """
    print(f"\n=== {mode_name} (matmul_precision={matmul_precision}, autocast_dtype={autocast_dtype}) ===",
          flush=True)
    torch.set_float32_matmul_precision(matmul_precision)
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )

    n_examples = len(vinson_data)
    total_batches = -(-n_examples // args.batch_size)  # ceil div
    num_warmup_batches = min(max(1, args.num_workers), total_batches)

    dataset = InferenceSequenceEmbedDataset(
        data=vinson_data,
        fasta_file=args.fasta_file,
        genotype_file=args.genotype_file,
        fasta_in_memory=args.fasta_in_memory,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == 'cuda',
        drop_last=False,
    )

    monitor = UtilizationMonitor(interval=args.poll_interval)
    preds = []
    with monitor, torch.no_grad(), autocast_ctx:
        loader_iter = iter(dataloader)
        load_start = time.monotonic()
        warmup_batches = [
            next(loader_iter)
            for _ in tqdm(range(num_warmup_batches), desc=f"{mode_name}: warmup")
        ]
        warmup_sec = time.monotonic() - load_start

        n_warmup_examples = 0
        for batch in warmup_batches:
            n_warmup_examples += len(batch['ohe_seq'])
            preds.append(model(batch['ohe_seq'].to(device), batch['embed'].to(device)).float().cpu().numpy())

        # Split "blocked waiting for next batch" (CPU: dataloader workers -- collate, one-hot
        # encoding, embedding lookup) from "forward pass" (GPU compute) to localize the
        # bottleneck. If data_wait_sec is near zero, workers are prefetching fast enough and
        # the GPU is the bottleneck; if it dominates, the CPU-side dataloader is.
        data_wait_sec = 0.0
        compute_sec = 0.0
        remaining_batches = total_batches - num_warmup_batches
        loop_start = time.monotonic()
        for _ in tqdm(range(remaining_batches), desc=f"{mode_name}: steady-state"):
            t0 = time.monotonic()
            batch = next(loader_iter)
            t1 = time.monotonic()
            preds.append(model(batch['ohe_seq'].to(device), batch['embed'].to(device)).float().cpu().numpy())
            t2 = time.monotonic()
            data_wait_sec += t1 - t0
            compute_sec += t2 - t1
        steady_state_sec = time.monotonic() - loop_start

    y_hat = np.concatenate(preds)
    remaining_examples = n_examples - n_warmup_examples
    examples_per_sec = remaining_examples / steady_state_sec if steady_state_sec > 0 else float("nan")

    result = {
        "mode": mode_name,
        "warmup_sec": warmup_sec,
        "steady_state_sec": steady_state_sec,
        "data_wait_sec": data_wait_sec,
        "compute_sec": compute_sec,
        "elapsed_sec": warmup_sec + steady_state_sec,
        "examples_per_sec": examples_per_sec,
        **monitor.summary(),
    }
    print(f"{mode_name}: warmup {warmup_sec:.1f}s ({num_warmup_batches} warmup batches), "
          f"steady-state {steady_state_sec:.1f}s (data_wait {data_wait_sec:.1f}s, compute {compute_sec:.1f}s), "
          f"{examples_per_sec:.1f} examples/sec", flush=True)
    return result, y_hat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark matmul precision settings (fp32/TF32/bf16 autocast) for DHS model inference."
    )
    parser.add_argument("anndata", type=str, help="Path to AnnData file (zarr-backed; read via read_zarr_backed)")
    parser.add_argument("fasta_file", type=str, help="Path to reference FASTA file")
    parser.add_argument("model_checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("model_config_path", type=str, help="Path to model config YAML file")
    parser.add_argument("--genotype_file", type=str, default=None, help="Path to TABIX indexed genotype file (optional)")
    parser.add_argument("--fasta_in_memory", action="store_true",
                         help="Use InMemoryFastaExtractor instead of disk-based FastaExtractor for every mode "
                              "in this run (fixed for the whole run, not itself compared across modes)")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=None,
                         help="Only benchmark the first N DHS regions (default: all)")
    parser.add_argument("--poll_interval", type=float, default=1.0, help="Seconds between CPU/GPU utilization samples")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path to write per-mode metrics as CSV")
    args = parser.parse_args()

    print(f"Device: {device}", flush=True)

    # var/obsm/obs_names are eager (only layers/varm are dask-backed by read_zarr_backed),
    # so no .compute() needed here.
    adata = read_zarr_backed(args.anndata)

    model_config = read_configs(args.model_config_path)
    model = dhs_model_from_config(
        model_config,
        checkpoint_path=args.model_checkpoint,
        device=device
    ).to(device).eval()

    sample_id = adata.obs_names[0]
    print(f"Using sample: {sample_id}", flush=True)

    bed_df = pd.DataFrame({
        "chrom": adata.var["#chr"].values,
        "summit": adata.var["dhs_summit"].values,
    })
    bed_df["start"] = bed_df["summit"]
    bed_df["end"] = bed_df["summit"] + 1

    vinson_data = VinsonData.inference_from_bed(
        bed_df,
        sample_ids=[sample_id],
        embeddings_df=adata.obsm["motif_embeddings"],
    )
    vinson_data = subset_data(vinson_data, args.max_examples)
    print(f"Benchmarking on {len(vinson_data)} examples", flush=True)

    results = []
    predictions = {}
    for mode_name, matmul_precision, autocast_dtype in MODES:
        result, y_hat = run_mode(mode_name, matmul_precision, autocast_dtype, vinson_data, model, args)
        results.append(result)
        predictions[mode_name] = y_hat

    # Reduced-precision modes are expected to differ slightly from the fp32 baseline -- this is a
    # sanity check that they're still close, not a strict equality check like the earlier
    # fasta-backend comparison was.
    baseline_name = MODES[0][0]
    baseline_preds = predictions[baseline_name]
    print(f"\n=== Prediction drift vs {baseline_name} ===")
    for mode_name, _, _ in MODES[1:]:
        max_diff = np.max(np.abs(predictions[mode_name] - baseline_preds))
        print(f"{mode_name}: max absolute difference = {max_diff:.6f}")

    print("\n=== Summary ===")
    fieldnames = ["mode", "warmup_sec", "steady_state_sec", "data_wait_sec", "compute_sec",
                  "elapsed_sec", "examples_per_sec", "ram_used_gb_mean", "ram_used_gb_max",
                  "gpu_mem_gb_mean", "gpu_mem_gb_max"]
    for row in results:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote metrics to {args.output_csv}")
