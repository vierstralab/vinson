"""
Compares GPU compute settings for DHS model inference, using the same dataloader/model path as
nextflow/bin/dhs/predict_DHS_model.py. Earlier profiling on this script (see git history) showed
data loading (fasta_disk vs InMemoryFastaExtractor) is not the bottleneck -- compute_sec is ~99%
of steady_state_sec -- so fasta backend is fixed for the whole run (disk by default; pass
--fasta_in_memory to use InMemoryFastaExtractor instead), not itself varied across modes here.

The trunk (LegNetTrunk, see vinson/models/sequence/legnet.py) is a depthwise-separable CNN (SE
blocks, grouped Conv1d, BatchNorm, residual concat) looped block-by-block in Python. That shape
informs the ladder below:
  - fp32_strict: true fp32 baseline (TF32 explicitly off for both matmul and cudnn -- cudnn's
    allow_tf32 defaults to True, so without this the "baseline" would already be doing TF32 convs).
  - cudnn_benchmark: + cudnn.benchmark=True. Seqlen/batch_size are fixed per run, so cuDNN can
    autotune the fastest conv algorithm for that exact shape -- free, precision-safe.
  - tf32: + real TF32 (matmul and cudnn). Depthwise convs have low arithmetic intensity (few FLOPs
    per byte moved) so are memory-bandwidth- rather than GEMM-bound -- expect this to matter less
    than it would for a dense/attention-heavy model.
  - tf32_compile: + torch.compile(model) on top of tf32. bf16 is excluded here -- bf16_autocast
    already showed large prediction drift (see max_abs_diff/max_rel_diff in earlier results) and
    isn't worth testing further even under compile. tf32_compile itself previously hung mid-run
    (grouped/depthwise Conv1d -- EffBlock's groups=self.inner_dim -- is a much less-tested path
    for Inductor's lowering/autotuning than dense/Conv2d patterns) even after forcing synchronous,
    non-pooled compilation via torch._inductor.config.compile_threads=1. Isolating it to this one
    mode, with bf16 out of the mix, to narrow down whether the hang is compile itself or specific
    to the tf32+compile combination.

If this also hangs: debugging a hung remote process with no attached profiler isn't a good
tradeoff against the gains already available without it (cudnn_benchmark is a free win; tf32 is
marginal). Drop tf32_compile from MODES and keep the first three.

Regions/sample are taken directly from the AnnData (no h5 export needed): every DHS
region (`adata.var`) is scored with the first sample's embedding (`adata.obs_names[0]`).
"""
import argparse
import contextlib
import csv
import threading
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch._inductor.config
from torch.utils.data import DataLoader
from tqdm import tqdm

from vinson.from_config import read_configs, dhs_model_from_config
from vinson.utils.data_formatting import VinsonData
from vinson.datasets.sequence import InferenceSequenceEmbedDataset

from genome_tools.data.anndata import read_zarr_backed

# Inductor's default async (multi-process) compile pool can deadlock on cleanup when repeatedly
# compiling the same model with torch._dynamo.reset() in between (one process, several modes) --
# seen in practice as a hang right after the last steady-state batch finishes. Forcing synchronous,
# in-process compilation avoids spawning that pool at all.
torch._inductor.config.compile_threads = 1

GB = 1024 ** 3


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# bf16 already ruled out (large prediction drift, see earlier max_abs_diff/max_rel_diff results)
# -- not worth testing further, including under compile. tf32_compile previously hung mid-run
# (grouped/depthwise Conv1d is a weak spot for Inductor's lowering/autotuning); isolating it to a
# single compile mode, with no bf16 variant in the mix, to narrow down whether it's compile itself
# or specifically the tf32+compile combination.
MODES = [
    {"name": "fp32_strict", "matmul_precision": "highest", "tf32": False, "cudnn_benchmark": False, "autocast_dtype": None, "compile": False},
    {"name": "cudnn_benchmark", "matmul_precision": "highest", "tf32": False, "cudnn_benchmark": True, "autocast_dtype": None, "compile": False},
    {"name": "tf32", "matmul_precision": "high", "tf32": True, "cudnn_benchmark": True, "autocast_dtype": None, "compile": False},
    {"name": "tf32_compile", "matmul_precision": "high", "tf32": True, "cudnn_benchmark": True, "autocast_dtype": None, "compile": True},
]


class UtilizationMonitor:
    """
    Samples RAM (GB, periodic) and this process's own GPU memory (GB) for the duration of a
    `with` block.

    GPU memory is read via torch.cuda's per-process accounting, not nvidia-smi: on a shared
    multi-GPU node (see the sbatch template's --gres=gpu:N), nvidia-smi reports every physical
    GPU on the node regardless of which one this job was actually allocated (CUDA_VISIBLE_DEVICES
    restricts the CUDA runtime, not nvidia-smi/NVML) -- a prior version of this monitor hit
    exactly that: gpu_mem_gb_max exceeding the card's actual capacity because another job's GPU on
    the same node got mixed into the samples.
    """

    def __init__(self, interval=1.0):
        self.interval = interval
        self.ram_used_gb = []
        self.gpu_mem_gb = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

    def _run(self):
        while not self._stop.is_set():
            self.ram_used_gb.append(psutil.virtual_memory().used / GB)
            if device == 'cuda':
                self.gpu_mem_gb.append(torch.cuda.memory_reserved() / GB)
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
        _, gpu_mem_mean = self._agg(self.gpu_mem_gb)
        # torch.cuda's own peak tracker is an exact continuous high-water-mark, unlike the sampled
        # max above which could miss a spike that happens between two poll ticks.
        gpu_mem_max = torch.cuda.max_memory_reserved() / GB if device == 'cuda' else float("nan")
        return {
            "ram_used_gb_mean": ram_mean, "ram_used_gb_max": ram_max,
            "gpu_mem_gb_mean": gpu_mem_mean, "gpu_mem_gb_max": gpu_mem_max,
        }


def subset_data(data: VinsonData, n: int) -> VinsonData:
    if n is None or n >= len(data):
        return data
    sliced = {key: value[:n] for key, value in data.data.items()}
    return VinsonData(sliced, data.encodings, data.embeddings_df, is_inference=data.is_inference)


def run_mode(mode_cfg, vinson_data, model, args):
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
    mode_name = mode_cfg["name"]
    print(f"\n=== {mode_name} ({mode_cfg}) ===", flush=True)

    torch.set_float32_matmul_precision(mode_cfg["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = mode_cfg["tf32"]
    torch.backends.cudnn.allow_tf32 = mode_cfg["tf32"]
    torch.backends.cudnn.benchmark = mode_cfg["cudnn_benchmark"]
    autocast_dtype = mode_cfg["autocast_dtype"]
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

    run_model = model
    if mode_cfg["compile"]:
        # torch.compile's first call pays a JIT trace+compile cost (seconds) -- trigger that here,
        # fully before the monitored/timed region, using a throwaway single-process batch of the
        # same shape as the real ones so it doesn't contaminate warmup_sec/steady_state_sec.
        # Dynamo's compiled-artifact cache is keyed to model.forward's code object + guards, not to
        # the torch.compile() wrapper instance -- since every mode here wraps the *same* underlying
        # model, a later mode (e.g. bf16_compile) could reuse an earlier mode's (e.g. tf32_compile)
        # cached graph if its guards don't fully capture the TF32/autocast change. Reset first so
        # each mode gets a clean compile reflecting its own current global settings.
        torch._dynamo.reset()
        run_model = torch.compile(model)
        warmup_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        compile_batch = next(iter(warmup_loader))
        with torch.no_grad(), autocast_ctx:
            for _ in range(2):  # a couple calls to settle any dynamic-shape guards
                run_model(compile_batch['ohe_seq'].to(device), compile_batch['embed'].to(device))
        if device == 'cuda':
            torch.cuda.synchronize()

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
            preds.append(run_model(batch['ohe_seq'].to(device), batch['embed'].to(device)).float().cpu().numpy())

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
            preds.append(run_model(batch['ohe_seq'].to(device), batch['embed'].to(device)).float().cpu().numpy())
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
        description="Benchmark GPU compute settings (fp32/cudnn.benchmark/TF32/TF32+compile) for DHS model inference."
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
    for mode_cfg in MODES:
        result, y_hat = run_mode(mode_cfg, vinson_data, model, args)
        results.append(result)
        predictions[mode_cfg["name"]] = y_hat

    # Reduced-precision modes are expected to differ slightly from the fp32 baseline -- this is a
    # sanity check that they're still close, not a strict equality check like the earlier
    # fasta-backend comparison was. Relative diff is normalized by |baseline| (eps-guarded against
    # near-zero baseline values, which would otherwise blow up an otherwise-tiny absolute diff).
    baseline_name = MODES[0]["name"]
    baseline_preds = predictions[baseline_name]
    eps = 1e-8
    for result in results:
        mode_name = result["mode"]
        if mode_name == baseline_name:
            result["max_abs_diff"] = result["mean_abs_diff"] = 0.0
            result["max_rel_diff"] = result["mean_rel_diff"] = 0.0
            continue
        abs_diff = np.abs(predictions[mode_name] - baseline_preds)
        rel_diff = abs_diff / (np.abs(baseline_preds) + eps)
        result["max_abs_diff"] = float(np.max(abs_diff))
        result["mean_abs_diff"] = float(np.mean(abs_diff))
        result["max_rel_diff"] = float(np.max(rel_diff))
        result["mean_rel_diff"] = float(np.mean(rel_diff))

    print(f"\n=== Prediction drift vs {baseline_name} ===")
    for result in results:
        print(f"{result['mode']}: max_abs={result['max_abs_diff']:.6f} mean_abs={result['mean_abs_diff']:.6f} "
              f"max_rel={result['max_rel_diff']:.6f} mean_rel={result['mean_rel_diff']:.6f}")

    print("\n=== Summary ===")
    fieldnames = ["mode", "warmup_sec", "steady_state_sec", "data_wait_sec", "compute_sec",
                  "elapsed_sec", "examples_per_sec", "max_abs_diff", "mean_abs_diff",
                  "max_rel_diff", "mean_rel_diff", "ram_used_gb_mean", "ram_used_gb_max",
                  "gpu_mem_gb_mean", "gpu_mem_gb_max"]
    for row in results:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote metrics to {args.output_csv}")
