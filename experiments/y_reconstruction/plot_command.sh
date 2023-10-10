# gmosa_1k
python make_statistical_comparisons.py\
    --results-table y_reconstruction/results_renamed.tsv\
    --outdir y_reconstruction/statistical_comparisons/bxt\
    --estimators\
        nrlmf\
        bxt_gso\
        bxt_gso__nrlmf\
        bxt_gso_1k\
        bxt_gso_1k__nrlmf\
        bxt_gmosa\
        bxt_gmosa__nrlmf\
        bxt_lmo\
        bxt_lmo__nrlmf\
        bxt_lso\
        bxt_lso__nrlmf\
        bxt_gmo\
        bxt_gmo__nrlmf\
&& python make_statistical_comparisons.py\
    --results-table y_reconstruction/results_renamed.tsv\
    --outdir y_reconstruction/statistical_comparisons/brf\
    --estimators\
        nrlmf\
        brf_gso\
        brf_gso__nrlmf\
        brf_gso_1k\
        brf_gso_1k__nrlmf\
        brf_gmosa\
        brf_gmosa__nrlmf\
        brf_lmo\
        brf_lmo__nrlmf\
        brf_lso\
        brf_lso__nrlmf

