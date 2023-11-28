python make_statistical_comparisons.py \
    --results-table prediction_weights/results.tsv \
    --outdir prediction_weights/statistical_comparisons/bxt \
    --estimators \
        bxt_gmosa bxt_gmo__softmax bxt_gmo__precomputed bxt_gmo__uniform \
        bxt_gmo__square bxt_gmo_full \
&& python make_statistical_comparisons.py \
    --results-table prediction_weights/results.tsv \
    --outdir prediction_weights/statistical_comparisons/brf \
    --estimators \
        brf_gmosa brf_gmo__softmax brf_gmo__precomputed brf_gmo__uniform \
        brf_gmo__square brf_gmo_full
