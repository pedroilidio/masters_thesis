#!/bin/bash

set -e

BASEDIR=y_reconstruction
METRICS=(
    "fit_time"
    "score_time"
    "fit_score_time"
    "LT+TL_average_precision"
    "LT+TL_roc_auc"
    "TT_average_precision"
    "TT_roc_auc"
)

python make_statistical_comparisons.py \
    --results-table $BASEDIR/results_renamed.tsv \
    --outdir $BASEDIR/statistical_comparisons/bxt \
    --metrics ${METRICS[@]} \
    --estimators \
        bxt_gso \
        bxt_gso__nrlmf \
        bxt_gmosa \
        bxt_gmosa__nrlmf \
        bxt_lmo \
        bxt_lmo__nrlmf \
        bxt_gmo \
        bxt_gmo__nrlmf

python make_statistical_comparisons.py \
    --results-table $BASEDIR/results_renamed.tsv \
    --outdir $BASEDIR/statistical_comparisons/brf \
    --metrics ${METRICS[@]} \
    --estimators \
        brf_gso \
        brf_gso__nrlmf \
        brf_gmosa \
        brf_gmosa__nrlmf \
        brf_lmo \
        brf_lmo__nrlmf \
        brf_gmo \
        brf_gmo__nrlmf
