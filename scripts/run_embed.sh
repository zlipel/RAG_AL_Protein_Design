#!/bin/bash
# Submit one embed job per PLM-compatible dataset.
# Run from the project root: bash scripts/run_embed.sh
#
# Each job precomputes all four PLM caches (mean, delta, site, physico) via
# submit_embed.sh, covering every PLM representation in the benchmark grid.
#
# BRCA1 is excluded — WT length 1863 exceeds ESM-2's 1022-residue limit.

set -eo pipefail

for D in \
    BLAT_ECOLX_Jacquier_2013 \
    BLAT_ECOLX_Deng_2012 \
    BLAT_ECOLX_Firnberg_2014 \
    BLAT_ECOLX_Stiffler_2015 \
    PABP_YEAST_Melamed_2013 \
    GFP_AEQVI_Sarkisyan_2016 \
    SPG1_STRSG_Wu_2016; do
    sbatch --job-name="embed_${D}" scripts/submit_embed.sh "$D"
done
