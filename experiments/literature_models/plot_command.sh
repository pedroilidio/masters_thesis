# gmosa_1k
# brf_gso_1k__nrlmf\
# bxt_gso_1k__nrlmf\
python make_statistical_comparisons.py \
    --results-table literature_models/results_renamed.tsv \
    --outdir literature_models/statistical_comparisons/ \
    --estimators\
        nrlmf\
        dnilmf\
        lmorls\
        blmnii_rls\
        blmnii_svm\
        dthybrid\
        md_ss_bxt_gso\
        bxt_gso_1k\
        bxt_gmosa_1k\
        bxt_gmosa__nrlmf\
        bxt_gso__nrlmf\
        bxt_gmo__nrlmf\
        brf_lmo\

        # bxt_lso\
        # bxt_gso\
        # bxt_lso__nrlmf\
        # bxt_gmo\
        # bxt_gmo__nrlmf\
        # bxt_gmosa\
        # bxt_lmo\
        # bxt_lmo__nrlmf\
