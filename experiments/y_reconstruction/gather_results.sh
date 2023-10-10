python make_runs_table.py --out y_reconstruction/results.tsv --runs y_reconstruction/runs bipartite_adaptations/runs \
&& python y_reconstruction/rename_estimators.py y_reconstruction/results.tsv
# Generate y_reconstruction/results_renamed.tsv
