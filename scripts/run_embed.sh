#!/bin/bash
# Submit one embed job per PLM-compatible dataset.
# Run from the project root: bash scripts/run_embed.sh
#
# Each job precomputes all four PLM caches (mean, delta, site, physico) via
# submit_embed.sh, covering every PLM representation in the benchmark grid.
#
# BRCA1 is excluded — WT length 1863 exceeds ESM-2's 1022-residue limit.
#
# GB1 (SPG1_STRSG_Wu_2016) is submitted separately with a longer wall time:
# ~149K variants × ~448 AA × 3 heavy forward passes (mean, site, physico) blows
# past submit_embed.sh's default 2h ceiling. Every other dataset fits in 2h —
# GFP, the next largest at ~51K × 238 AA, comfortably so.

set -eo pipefail

# --- Standard datasets: default 2h wall time (from submit_embed.sh) -----------
for D in \
    BLAT_ECOLX_Jacquier_2013 \
    BLAT_ECOLX_Deng_2012 \
    BLAT_ECOLX_Firnberg_2014 \
    BLAT_ECOLX_Stiffler_2015 \
    PABP_YEAST_Melamed_2013 \
    GFP_AEQVI_Sarkisyan_2016; do
    sbatch --job-name="embed_${D}" scripts/submit_embed.sh "$D"
done

# --- GB1: large dataset, override wall time --------------------------------
# `sbatch --time=...` on the command line overrides the #SBATCH --time baked
# into submit_embed.sh. Raise GB1_WALLTIME if the job still hits the limit.
GB1_WALLTIME="${GB1_WALLTIME:-08:00:00}"
sbatch --time="$GB1_WALLTIME" --job-name="embed_SPG1_STRSG_Wu_2016" \
    scripts/submit_embed.sh SPG1_STRSG_Wu_2016
