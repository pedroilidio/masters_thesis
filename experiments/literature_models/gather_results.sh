# Generate literature_models/results_renamed.tsv
python make_runs_table.py \
    --out literature_models/results.tsv \
    --runs y_reconstruction/runs bipartite_adaptations/runs literature_models/runs \
&& python y_reconstruction/rename_estimators.py literature_models/results.tsv
