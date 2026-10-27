"""
Compares FastaExtractor (disk, re-read per example) vs InMemoryFastaExtractor (whole genome
loaded once) for DHS model inference, using the same dataloader/model path as
nextflow/bin/dhs/predict_DHS_model.py. Reports wall time, throughput, CPU/RAM and GPU utilization
for each, and checks that both backends produce identical predictions.

Regions/sample are taken directly from the training AnnData (no h5 export needed): every DHS
region (`adata.var`) is scored against the first sample (`adata.obs_names[0]`).

Note: with --num_workers > 0, InMemoryFastaExtractor is loaded once per worker process (each
worker gets its own copy), so peak RAM scales with num_workers for the in-memory run.
"""
import argparse
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

from vinson.from_config import read_configs, dhs_model_from_config
from vinson.utils.data_formatting import VinsonData
from vinson.inference import score_bed_with_dhs_model
from vinson.datasets.sequence import InferenceSequenceEmbedDataset

from genome_tools.data.anndata import read_zarr_backed


device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODES = [
    ("fasta_disk", False),
    ("fasta_in_memory", True),
]


class UtilizationMonitor:
    """Samples CPU/RAM/GPU usage in a background thread for the duration of a `with` block."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.cpu_pct = []
        self.ram_pct = []
        self.gpu_util_pct = []
        self.gpu_mem_mb = []
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _poll_gpu(self):
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        for line in out.strip().splitlines():
            util, mem = line.split(",")
            self.gpu_util_pct.append(float(util))
            self.gpu_mem_mb.append(float(mem))

    def _run(self):
        psutil.cpu_percent(percpu=False)  # prime; first call always returns 0.0
        while not self._stop.is_set():
            self.cpu_pct.append(psutil.cpu_percent(percpu=False))
            self.ram_pct.append(psutil.virtual_memory().percent)
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
        cpu_max, cpu_mean = self._agg(self.cpu_pct)
        ram_max, ram_mean = self._agg(self.ram_pct)
        gpu_util_max, gpu_util_mean = self._agg(self.gpu_util_pct)
        gpu_mem_max, gpu_mem_mean = self._agg(self.gpu_mem_mb)
        return {
            "cpu_pct_mean": cpu_mean, "cpu_pct_max": cpu_max,
            "ram_pct_mean": ram_mean, "ram_pct_max": ram_max,
            "gpu_util_pct_mean": gpu_util_mean, "gpu_util_pct_max": gpu_util_max,
            "gpu_mem_mb_mean": gpu_mem_mean, "gpu_mem_mb_max": gpu_mem_max,
        }


def subset_data(data: VinsonData, n: int) -> VinsonData:
    if n is None or n >= len(data):
        return data
    sliced = {key: value[:n] for key, value in data.data.items()}
    return VinsonData(sliced, data.encodings, data.embeddings_df, is_inference=data.is_inference)


def run_mode(mode_name, fasta_in_memory, vinson_data, model, args):
    print(f"\n=== {mode_name} (fasta_in_memory={fasta_in_memory}) ===", flush=True)

    monitor = UtilizationMonitor(interval=args.poll_interval)
    start = time.monotonic()
    with monitor:
        dataset = InferenceSequenceEmbedDataset(
            data=vinson_data,
            fasta_file=args.fasta_file,
            genotype_file=args.genotype_file,
            fasta_in_memory=fasta_in_memory,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device == 'cuda',
            drop_last=False,
        )
        y_hat = score_bed_with_dhs_model(dataloader, model, device=device)
    elapsed = time.monotonic() - start

    n_examples = len(vinson_data)
    result = {
        "mode": mode_name,
        "elapsed_sec": elapsed,
        "examples_per_sec": n_examples / elapsed,
        **monitor.summary(),
    }
    print(f"{mode_name}: {elapsed:.1f}s, {result['examples_per_sec']:.1f} examples/sec", flush=True)
    return result, y_hat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark FastaExtractor vs InMemoryFastaExtractor for DHS model inference."
    )
    parser.add_argument("anndata", type=str, help="Path to AnnData file (zarr-backed; read via read_zarr_backed)")
    parser.add_argument("fasta_file", type=str, help="Path to reference FASTA file")
    parser.add_argument("model_checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("model_config_path", type=str, help="Path to model config YAML file")
    parser.add_argument("--genotype_file", type=str, default=None, help="Path to TABIX indexed genotype file (optional)")

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
    ).eval()

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
    print(vinson_data.to_df())
    print(f"Benchmarking on {len(vinson_data)} examples", flush=True)

    results = []
    predictions = {}
    for mode_name, fasta_in_memory in MODES:
        result, y_hat = run_mode(mode_name, fasta_in_memory, vinson_data, model, args)
        results.append(result)
        predictions[mode_name] = y_hat

    disk_preds, memory_preds = predictions["fasta_disk"], predictions["fasta_in_memory"]
    matches = np.allclose(disk_preds, memory_preds)
    print(f"\nPredictions match between backends: {matches}", flush=True)
    if not matches:
        max_diff = np.max(np.abs(disk_preds - memory_preds))
        print(f"WARNING: max absolute difference between backends = {max_diff}", flush=True)

    print("\n=== Summary ===")
    fieldnames = ["mode", "elapsed_sec", "examples_per_sec", "cpu_pct_mean", "cpu_pct_max",
                  "ram_pct_mean", "ram_pct_max", "gpu_util_pct_mean", "gpu_util_pct_max",
                  "gpu_mem_mb_mean", "gpu_mem_mb_max"]
    for row in results:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote metrics to {args.output_csv}")
