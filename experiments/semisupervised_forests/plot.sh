python make_statistical_comparisons.py \
    --results-table semisupervised_forests/results_renamed.tsv \
    --outdir semisupervised_forests/statistical_comparisons/no_drop \
    --metrics \
        LT+TL_average_precision \
        LT+TL_roc_auc \
        TT_average_precision \
        TT_roc_auc \
        fit_time \
        fit_score_time \
    --estimators \
        ad_fixed \
        ad_density \
        ad_size \
        ad_random \
        md_fixed \
        md_density \
        md_size \
        md_random \
        mse_fixed \
        mse_density \
        mse_size \
        mse_random \
&& python make_statistical_comparisons.py \
    --results-table semisupervised_forests/results_renamed.tsv \
    --outdir semisupervised_forests/statistical_comparisons/drop50 \
    --metrics \
        LT+TL_average_precision \
        LT+TL_roc_auc \
        TT_average_precision \
        TT_roc_auc \
        fit_time \
        fit_score_time \
    --estimators \
        ad_fixed__50 \
        ad_density__50 \
        ad_size__50 \
        ad_random__50 \
        md_fixed__50 \
        md_density__50 \
        md_size__50 \
        md_random__50 \
        mse_fixed__50 \
        mse_density__50 \
        mse_size__50 \
        mse_random__50 \
&& python make_statistical_comparisons.py \
    --results-table semisupervised_forests/results_renamed.tsv \
    --outdir semisupervised_forests/statistical_comparisons/drop70 \
    --metrics \
        LT+TL_average_precision \
        LT+TL_roc_auc \
        TT_average_precision \
        TT_roc_auc \
        fit_time \
        fit_score_time \
    --estimators \
        ad_fixed__70 \
        ad_density__70 \
        ad_size__70 \
        ad_random__70 \
        md_fixed__70 \
        md_density__70 \
        md_size__70 \
        md_random__70 \
        mse_fixed__70 \
        mse_density__70 \
        mse_size__70 \
        mse_random__70 \
&& python make_statistical_comparisons.py \
    --results-table semisupervised_forests/results_renamed.tsv \
    --outdir semisupervised_forests/statistical_comparisons/drop90 \
    --metrics \
        LT+TL_average_precision \
        LT+TL_roc_auc \
        TT_average_precision \
        TT_roc_auc \
        fit_time \
        fit_score_time \
    --estimators \
        ad_fixed__90 \
        ad_density__90 \
        ad_size__90 \
        ad_random__90 \
        md_fixed__90 \
        md_density__90 \
        md_size__90 \
        md_random__90 \
        mse_fixed__90 \
        mse_density__90 \
        mse_size__90 \
        mse_random__90
