import os
import logging

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from genome_tools import GenomicInterval as genomic_interval
from genome_tools.data.extractors import BigwigExtractor as bigwig_extractor

from vinson.utils.helpers import replace_at
from vinson.utils.sequence_utils import one_hot_encode
from .plotting import LoggerMixin 

from matplotlib import colors as mcolors



class VariantDataPreprocessor(LoggerMixin):
    def __init__(
        self,
        adata,
        group_file_locations,
        model=None,
        fasta_extractor=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.adata = adata
        self.model = model
        self.fasta_extractor = fasta_extractor
        self.metadata = adata.obs
        self.embeddings = adata.obsm["motif_embeddings"]

        self.metadata["group_name"] = (
            self.metadata["extended_annotation"]
            .str.replace(" ", "_", regex=False)
            .str.replace("/", "_", regex=False)
        )

        # group -> parquet path lookup
        self.group_file_locations = (
            pd.read_csv(group_file_locations, sep="\t")
            .set_index("group_id")["file_location"]
            .to_dict()
        )
        parquet_groups = set(self.group_file_locations.keys())
        metadata_groups = set(self.metadata["group_name"].dropna().unique())

        groups = sorted(self.metadata["group_name"].dropna().unique())

        cmap = plt.get_cmap("tab20")

        self.names_to_colors = {
            group: mcolors.to_hex(cmap(i % cmap.N))
            for i, group in enumerate(groups)
        }

    def get_group_embeddings(self, group_ids):

        mask = self.metadata["group_name"].isin(group_ids)
    
        # make sure embeddings are numpy
        embeddings = np.asarray(self.embeddings)
    
        groups = self.metadata.loc[mask, "group_name"].values
    
        embeddings = embeddings[mask.values]
    
        group_embeddings = {}
    
        for group in group_ids:
    
            group_mask = groups == group
    
            group_embeddings[group] = torch.tensor(
                embeddings[group_mask],
                dtype=torch.float32
            )
    
        return group_embeddings
    def get_prediction_df(self, variant_id, group_ids=None):
    
        chrom, pos, ref, alt = variant_id.split(":")
        pos = int(pos)
    
        dfs = []
    
        for group_id, path in self.group_file_locations.items():
    
            if group_ids is not None and group_id not in group_ids:
                continue
    
            df = pd.read_parquet(path)
    
            mask = (
                (df["#chr"] == chrom) &
                (df["end"] == pos) &
                (df["ref"] == ref) &
                (df["alt"] == alt)
            )
            
            variant_df = df.loc[mask]

            if variant_df.empty:
                continue
            
            dfs.append(variant_df)
    
        if not dfs:
            return pd.DataFrame()
    
        pred_df = pd.concat(dfs, ignore_index=True)
        pred_df = pred_df.rename(
            columns={"group_id": "group_name"}
        )
    
        return pred_df
    
    
    
    def aggregate_predictions(self, pred_df):
        # filter bad predictions/variants first
        pred_df = pred_df[
            pred_df["hotspots"].astype(str).isin(["1", "-"])
        ].copy()
    
        summary = (
            pred_df
            .groupby("group_name")
            .apply(
                lambda x: pd.Series({
        
                    "#chr": x["#chr"].iloc[0],
                    "end": x["end"].iloc[0],
                    "ref": x["ref"].iloc[0],
                    "alt": x["alt"].iloc[0],
        
                    "logit_group_es": np.average(
                        x["logit_es"],
                        weights=x["inverse_mse"]
                    ),
                    
                    "logit_es_std": np.sqrt(
                        np.average(
                            (
                                x["logit_es"] -
                                np.average(
                                    x["logit_es"],
                                    weights=x["inverse_mse"]
                                )
                            )**2,
                            weights=x["inverse_mse"]
                        )
                    ),
        
                    "pred_logit_es": np.average(
                        x["pred"],
                        weights=x["inverse_mse"]
                    ),
        
                    "pred_logit_es_std": np.sqrt(
                        np.average(
                            (
                                x["pred"] -
                                np.average(
                                    x["pred"],
                                    weights=x["inverse_mse"]
                                )
                            )**2,
                            weights=x["inverse_mse"]
                        )
                    ),
        
                    "n_samples": len(x)
        
                }),
                include_groups=False
            )
            .reset_index()
        )
        
        summary["variant_id"] = (
            summary["#chr"].astype(str)
            + ":"
            + summary["end"].astype(str)
            + ":"
            + summary["ref"]
            + ":"
            + summary["alt"]
        )
        summary["pred_logit_es"] = summary["pred_logit_es"] / np.log(2)
        summary["pred_logit_es_std"] = summary["pred_logit_es_std"] / np.log(2)
    
        return summary
    def get_interval(self, variant_id, width=2000):
    
        chrom, pos, ref, alt = variant_id.split(":")
        pos = int(pos)
    
        interval = genomic_interval(
            chrom,
            pos,
            pos
        ).widen(width)
    
        return interval
        
    def get_density(self, variant_id, tested_df, group_ids=None):
        
        interval = self.get_interval(variant_id, width=2000)
    
        metadata = self.metadata

        # Add this variant's sample-level values
        metadata = metadata.join(
            tested_df[["logit_es", "inverse_mse"]]
        )
        
        # Keep only samples where this variant is present/tested
        metadata = metadata[
            metadata["logit_es"] != 0
        ]
        
        if group_ids is not None:
            metadata = metadata[
                metadata["group_name"].isin(group_ids)
            ]
    
        dens = {}
    
        for group_id, df in metadata.groupby("group_name"):
    
            bw_files = df["normalized_density_bw"].dropna()
    
            if len(bw_files) == 0:
                continue
    
            signals = []
    
            for bw in bw_files:
                bw_extractor = bigwig_extractor(bw)
                signals.append(bw_extractor[interval])

            weights = tested_df.loc[df.index, "inverse_mse"].to_numpy()
            
            dens[group_id] = np.average(
                np.array(signals),
                axis=0,
                weights=weights
            )
    
        return dens, interval
    def get_variant_sequences(
        self,
        variant_id,
        interval,
        fasta_extr
        
    ):
        chrom, pos, ref, alt = variant_id.split(":")
        pos = int(pos)
    
        # get reference sequence
        dna_seq = fasta_extr[interval]
    
        # variant position relative to interval
        rel = pos - interval.start
        start = pos-1 
        end = pos
        rel = start - interval.start
        dna_seq_ref = replace_at(dna_seq, rel, ref)
        dna_seq_alt = replace_at(dna_seq, rel, alt)
        # convert to one-hot
        seq_ref = one_hot_encode(dna_seq_ref)
        seq_alt = one_hot_encode(dna_seq_alt)
    
        return seq_ref, seq_alt

    def get_variant_shap(
        self,
        variant_id,
        model,
        interval,
        group_embeddings,
        fasta_extractor,
        seqlen=1344,
        device='cpu',
    ):
        
        seq_ref, seq_alt = self.get_variant_sequences(
            variant_id,
            interval,
            fasta_extractor,
        )
        
        seq_ref = torch.tensor(
            seq_ref,
            dtype=torch.float32,
            device=device
        )
        
        seq_alt = torch.tensor(
            seq_alt,
            dtype=torch.float32,
            device=device
        )
        
        if seq_ref.ndim == 2:
            seq_ref = seq_ref.unsqueeze(0)
        
        if seq_alt.ndim == 2:
            seq_alt = seq_alt.unsqueeze(0)
    
        # ---------------------------
        # Calculate SHAP
        # ---------------------------
        shap_scores = {}
    
        for group, embeddings in group_embeddings.items():
    
            embeddings = embeddings.to(device)
        
            # average embeddings first
            mean_embed = embeddings.mean(dim=0, keepdim=True)
        
            attrs = model.get_variant_sequence_attributions(
                ref_seq=seq_ref,
                alt_seq=seq_alt,
                embed=mean_embed,
                device=device,
                hypothetical=True,
            )
    
            ref_attr = attrs["ref"].detach().cpu().numpy()
            alt_attr = attrs["alt"].detach().cpu().numpy()
            
            # (1,4,1344) -> (1344,4)
            ref_attr = ref_attr.squeeze(0).T
            alt_attr = alt_attr.squeeze(0).T
            
            # convert one-hot sequences to (1344,4)
            ref_seq = seq_ref.squeeze(0).cpu().numpy().T
            alt_seq = seq_alt.squeeze(0).cpu().numpy().T
            
            # keep only the observed base at each position
            ref_attr *= ref_seq
            alt_attr *= alt_seq
            
            shap_scores[group] = {
                "ref": ref_attr,
                "alt": alt_attr,
            }
    
        return shap_scores
        
    def get_variant_data(self, variant_id, components,group_ids=None):

        pred_df = self.get_prediction_df(
            variant_id,
            group_ids=group_ids
        )
        
        anova_df = self.aggregate_predictions(pred_df)
    
        tested_df = self.get_tested_df(variant_id)

        tested_groups = (
            tested_df.loc[
                tested_df["logit_es"] != 0,
                "group_name"
            ]
            .dropna()
            .unique()
        )
        
        if group_ids is None:
            group_ids = tested_groups
        else:
            group_ids = np.intersect1d(
                np.array(group_ids, dtype=str),
                np.array(tested_groups, dtype=str)
            )
        anova_df = anova_df[
            anova_df["group_name"].isin(group_ids)
        ]
        
        dens, interval = self.get_density(
            variant_id,
            tested_df=tested_df,
            group_ids=group_ids
        )
        shap_interval = self.get_interval(variant_id, width=672)

        density_groups = np.array(list(dens.keys()))
        
        # keep only groups that have both
        anova_df = anova_df[
            anova_df["group_name"].isin(density_groups)
        ].copy()
        
        tested_df = tested_df[
            tested_df["group_name"].isin(density_groups)
        ].copy()

        shap_scores = {}

        anova_df = (
            anova_df
            .sort_values("logit_group_es")
            .reset_index(drop=True)
        )
        tested_df["group_name"] = pd.Categorical(
            tested_df["group_name"],
            categories=anova_df["group_name"],
            ordered=True,
        )
        
        tested_df = tested_df.sort_values("group_name")
    
        component_data = []
    
        for component in components:
    
            if component.__class__.__name__ == "DensityPlotComponent":
                component_data.append(
                    (anova_df, dens, interval)
                )
    
            elif component.__class__.__name__ == "ImbalancePlotComponent":
                component_data.append(
                    (tested_df, anova_df)
                )
            elif component.__class__.__name__ == "PredictionImbalancePlotComponent":
                component_data.append(anova_df)
    
            elif component.__class__.__name__ == "PredictionPlotComponent":
                component_data.append(
                    anova_df
                )
            elif component.__class__.__name__ == "ContributionPlotComponent":
                if (
                    self.model is None
                    or self.fasta_extractor is None
                ):
                    raise ValueError(
                        "ContributionPlotComponent requires "
                        "model and fasta_extractor"
                    )
            
                # groups that actually appear in this plot
                group_embeddings = self.get_group_embeddings(
                    density_groups
                )
            
                shap_path = None
            
                shap_scores = self.get_variant_shap(
                    variant_id=variant_id,
                    model=self.model,
                    interval=shap_interval,
                    group_embeddings=group_embeddings,
                    fasta_extractor=self.fasta_extractor,
                )
            
            
                component_data.append(
                    (
                        anova_df,
                        shap_scores
                    )
                )
    
        return components, component_data, len(anova_df)
        
    def get_tested_df(self, variant_id):

        chrom, pos, ref, alt = variant_id.split(":")
        pos = int(pos)
    
        var = self.adata.var
    
        idx = (
            (var["#chr"] == chrom) &
            (var["end"] == pos) &
            (var["ref"] == ref) &
            (var["alt"] == alt)
        )
    
        var_idx = np.where(idx)[0][0]
    
        logit_es = np.array(
            self.adata.layers["logit_es"][:, var_idx].todense()
        ).flatten()

        inverse_mse = np.array(
            self.adata.layers["inverse_mse"][:, var_idx].todense()
        ).flatten()
    
        tested_df = self.metadata[["group_name"]].copy()

        tested_df["sample_id"] = tested_df.index
        tested_df["logit_es"] = logit_es
        tested_df["inverse_mse"] = inverse_mse
        
        return tested_df
    


    