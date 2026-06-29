#!/bin/bash
#SBATCH --job-name=prep_filter_phased_genotypes
#SBATCH --partition=hpcz-test
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/home/mbrannon/tmp/logs/prep_filter_%j.out
#SBATCH --error=/home/mbrannon/tmp/logs/prep_filter_%j.err

set -euo pipefail

# -------------------------------------------------
# INPUT FILES
# -------------------------------------------------

ATAC_IN=/net/seq/data2/projects/nasi4/ENCODE4/atac-genotypes.v6/round2/output/all_phased.bed.gz
DNASE_IN=/net/seq/data2/projects/nasi4/ENCODE4/dnase-genotypes.v5/round2/output/all_phased.bed.gz

# -------------------------------------------------
# WORK DIRECTORY
# -------------------------------------------------

WORKDIR=/home/mbrannon/tmp/filter_variants_${SLURM_JOB_ID}
mkdir -p ${WORKDIR}
cd ${WORKDIR}

echo "[INFO] Working directory: ${WORKDIR}"

# -------------------------------------------------
# STEP 1: Create genotype files with prefixes
# -------------------------------------------------

GT_ATAC=${WORKDIR}/genotype.atac.phased.tsv
GT_DNASE=${WORKDIR}/genotype.dnase.phased.tsv

echo "[INFO] Processing DNASE phased genotype"

zcat ${DNASE_IN} \
| awk 'BEGIN{FS=OFS="\t"} {$6="DNASE_"$6; print}' \
> ${GT_DNASE}

echo "[INFO] Processing ATAC phased genotype"

zcat ${ATAC_IN} \
| awk 'BEGIN{FS=OFS="\t"} {$6="ATAC_"$6; print}' \
> ${GT_ATAC}

# -------------------------------------------------
# Function to remove multiallelic sites
# -------------------------------------------------

clean_genotype () {

    infile=$1
    outfile=$2
    sorted=${outfile}.sorted

    echo "[INFO] Cleaning ${infile}"

    sort \
        -k1,1 \
        -k2,2n \
        -k6,6 \
        --parallel=${SLURM_CPUS_PER_TASK} \
        --buffer-size=4G \
        "${infile}" \
        > "${sorted}"

    awk '
    BEGIN{
        FS=OFS="\t"
    }

    {
        site=$1 FS $2 FS $6
        allele=$4 FS $5

        if(site != prev_site && NR > 1){

            if(length(alleles)==1){
                for(i=1;i<=n;i++)
                    print lines[i]
            }

            delete alleles
            delete lines
            n=0
        }

        alleles[allele]=1

        n++
        lines[n]=$0

        prev_site=site
    }

    END{

        if(length(alleles)==1){

            for(i=1;i<=n;i++)
                print lines[i]

        }

    }
    ' "${sorted}" > "${outfile}"

    rm "${sorted}"
}

# -------------------------------------------------
# STEP 2: Remove multiallelic sites
# -------------------------------------------------

clean_genotype \
    "${GT_ATAC}" \
    "${WORKDIR}/atac.phased.clean.tsv"

clean_genotype \
    "${GT_DNASE}" \
    "${WORKDIR}/dnase.phased.clean.tsv"

echo "[INFO] Finished cleaning genotype files"

echo
echo "[INFO] Outputs:"
echo "${WORKDIR}/atac.phased.clean.tsv"
echo "${WORKDIR}/dnase.phased.clean.tsv"
