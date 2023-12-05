#!/bin/bash
set -e

BASEDIR=literature_models
METRICS=(
    "fit_time"
    "score_time"
    "fit_score_time"
    "LT+TL_average_precision"
    "LT+TL_roc_auc"
    "TT_average_precision"
    "TT_roc_auc"
)
ESTIMATORS=(
    "nrlmf"
    "dnilmf"
    "lmo_rls"
    "kron_rls"
    "mlp"
    "blmnii_rls"
    "blmnii_svm"
    "dthybrid"
    "bxt_sgso_us"
    "bxt_gso_1k"
    "bxt_gmosa_1k"
    "bxt_gmosa__nrlmf"
    "bxt_gso__nrlmf"
    "bxt_gmo__nrlmf"
    "brf_lmo"
    "ss_bxt_gso__md_size"
    "ss_bxt_gso__ad_fixed"
    "ss_bxt_gso__mse_density"
)

echo "*** NO DROP ***"
python make_statistical_comparisons.py \
    --results-table $BASEDIR/results_renamed.tsv \
    --outdir $BASEDIR/statistical_comparisons/no_drop \
    --estimators ${ESTIMATORS[@]} \
    --metrics ${METRICS[@]}

echo "*** DROP 50% ***"
python make_statistical_comparisons.py \
    --results-table $BASEDIR/results_renamed.tsv \
    --outdir $BASEDIR/statistical_comparisons/drop50 \
    --estimators $(for E in ${ESTIMATORS[@]}; do echo $E"__50"; done) \
    --metrics ${METRICS[@]}

echo "*** DROP 70% ***"
python make_statistical_comparisons.py \
    --results-table $BASEDIR/results_renamed.tsv \
    --outdir $BASEDIR/statistical_comparisons/drop70 \
    --estimators $(for E in ${ESTIMATORS[@]}; do echo $E"__70"; done) \
    --metrics ${METRICS[@]}

echo "*** DROP 90% ***"
python make_statistical_comparisons.py \
    --results-table $BASEDIR/results_renamed.tsv \
    --outdir $BASEDIR/statistical_comparisons/drop90 \
    --estimators $(for E in ${ESTIMATORS[@]}; do echo $E"__90"; done) \
    --metrics ${METRICS[@]}

