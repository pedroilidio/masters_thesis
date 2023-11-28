python make_statistical_comparisons.py \
    --results-table bipartite_adaptations/results.tsv \
    --outdir bipartite_adaptations/statistical_comparisons/brf \
    --estimators \
        brf_gmo brf_gmosa brf_lmo brf_lso brf_gso brf_sgso_us \
&& python make_statistical_comparisons.py \
    --results-table bipartite_adaptations/results.tsv \
    --outdir bipartite_adaptations/statistical_comparisons/bxt \
    --estimators \
        bxt_gmo bxt_gmosa bxt_lmo bxt_lso bxt_gso bxt_sgso_us \
&& python make_statistical_comparisons.py \
    --results-table bipartite_adaptations/results.tsv \
    --outdir bipartite_adaptations/statistical_comparisons/gso \
    --estimators \
        bxt_gso bxt_sgso bxt_sgso_us \
        bxt_gso bxt_sgso bxt_sgso_us \
