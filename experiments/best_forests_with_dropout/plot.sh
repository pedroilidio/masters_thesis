#!/bin/bash
set -e

DEBUG=$1

BASEDIR=best_forests_with_dropout
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
    "bxt_gso"
    # "bxt_gmo"
    "bxt_gmosa"
    # "bxt_gso_1k"
    # "bxt_gmosa_1k"
    "bxt_gso__nrlmf"
    "bxt_gmosa__nrlmf"
    "bxt_gmo__nrlmf"  # FIXME
    "brf_lmo"
    "ss_bxt_gso__md_size"
    "ss_bxt_gso__ad_fixed"
    "ss_bxt_gso__mse_density"
)

# gmosa_1k
# brf_gso_1k__nrlmf\
# bxt_gso_1k__nrlmf\

echo "*** NO DROP ***"
python $DEBUG make_statistical_comparisons.py \
    --results-table $BASEDIR/results_renamed.tsv \
    --outdir $BASEDIR/statistical_comparisons/no_drop \
    --estimators ${ESTIMATORS[@]} \
    --metrics ${METRICS[@]} \
    --raise-missing

for drop in 50 70 90; do
    echo "*** DROP $drop% ***"
    python $DEBUG make_statistical_comparisons.py \
        --results-table $BASEDIR/results_renamed.tsv \
        --outdir $BASEDIR/statistical_comparisons/drop$drop \
        --estimators $(for E in ${ESTIMATORS[@]}; do echo $E"__"$drop; done) \
        --metrics ${METRICS[@]} \
        --raise-missing
done