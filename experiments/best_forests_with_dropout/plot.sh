#!/bin/bash
# brf_gso_1k__nrlmf\
# bxt_gso_1k__nrlmf\
#        bxt_lso\
#        bxt_lso__nrlmf\
#        brf_lso\
#        brf_lso__nrlmf
#        bxt_gso_1k\
#        bxt_gmosa_1k\
#        brf_gso_1k\
#        brf_gmosa_1k\
#        nrlmf\

python make_statistical_comparisons.py\
    --results-table y_reconstruction/results_renamed.tsv\
    --outdir y_reconstruction/statistical_comparisons/best_models\
    --estimators\
        bxt_gmosa__nrlmf\
        bxt_gmo__nrlmf\
        brf_gmo__nrlmf\
        brf_gso__nrlmf\
        brf_lmo\
        brf_gmo\
        bxt_gso__nrlmf\
        bxt_gmo

