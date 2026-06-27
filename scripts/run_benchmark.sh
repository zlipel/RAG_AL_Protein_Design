#!/bin/bash
# Submit one benchmark job per dataset (all representations × acquisitions × seeds).
# Run from the project root: bash scripts/run_benchmark.sh
#
# Prerequisites: run_embed.sh jobs must have completed first for PLM representations.

set -eo pipefail

for D in \
    BLAT_ECOLX_Jacquier_2013 \
    BLAT_ECOLX_Deng_2012 \
    BLAT_ECOLX_Firnberg_2014 \
    BLAT_ECOLX_Stiffler_2015 \
    PABP_YEAST_Melamed_2013 \
    BRCA1_HUMAN_Findlay_2018; do
    sbatch --job-name="bench_${D}" scripts/submit_benchmark.sh "$D"
done
