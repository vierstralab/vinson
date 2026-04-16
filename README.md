# vinson  
Machine learning sequence-to-function models for predicting DNA accessibility (DHSs) and variant effects.

---

## Overview

This repository contains two main components:

1. **DHS (sequence-to-function) prediction models** via Nextflow pipelines  
2. **Variant effect prediction models** using AnnData-based training and PyTorch inference  

---

# 1. Variant Effect Modeling Pipeline

This workflow trains and evaluates models that predict variant effects from sequence and genomic context.

---

## Step 1: Create AnnData Training Dataset

Use the notebook: creating_adata.ipynb


### Purpose:
- Takes a **variant file** (parquet format)
- Takes a **genotype file**
- Filters variants (coverage, BAD score, etc.)
- Builds a structured **AnnData object**
  - variant features stored in `.var`
  - sample observations stored in `.obs`
  - numeric signals stored in `.layers`
  - embeddings stored in `.obsm`
- Outputs `.h5ad` file used for model training

---

## Step 2: Train Variant Model (SBATCH)

Submit training job using: submit_to_sbatch.py example in bottom of creating_adata.ipynb
- check training stats in this notebook check_training.ipynb

---

# 3. Variant Prediction and Analysis

After training, use Predict_variants.ipynb, has validation dataset prediction, single sample prediciton, variant specific prediction, prediciton pipeline code, prediction aggregation, cell type effect size plotting

---

## Notes

- WIP: add predicitons to nextflow pipeline, add more plotting visualization code and contribution plots 