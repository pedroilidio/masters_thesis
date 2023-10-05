from time import perf_counter
import numpy as np
import pandas as pd
from bipartite_learn.tree import (
    BipartiteDecisionTreeRegressor,
    BipartiteExtraTreeRegressor,
)
from sklearn.utils import check_random_state
# from make_examples import make_interaction_regression

SEED = 0
OUTPATH = "fit_time_bxt_bdt.csv"


def main(outpath=OUTPATH, random_state=SEED):
    random_state = check_random_state(random_state)

    estimators = {
        "bxt_gso": BipartiteExtraTreeRegressor(
            random_state=random_state,
            criterion="squared_error_gso",
            bipartite_adapter="gmosa",
        ),
        "bdt_gso": BipartiteDecisionTreeRegressor(
            random_state=random_state,
            criterion="squared_error_gso",
            bipartite_adapter="gmosa",
        ),
        "bxt_gmo": BipartiteExtraTreeRegressor(
            random_state=random_state,
            criterion="squared_error",
            bipartite_adapter="gmosa",
        ),
        "bdt_gmo": BipartiteDecisionTreeRegressor(
            random_state=random_state,
            criterion="squared_error",
            bipartite_adapter="gmosa",
        ),
    }

    nn = np.logspace(2, 4, 50, dtype=int)
    records = []

    for i, n in enumerate(nn):
        print(f"Starting n={n} ({i + 1}/{len(nn)})")
        # X, y = make_interaction_regression(
        #     n_samples=(n, n),
        #     n_features=(n, n),
        #     noise=0.0,
        #     random_state=random_state,
        # )
        *X, y = random_state.random((3, n, n))

        # Train and time estimators
        for estimator_name, estimator in estimators.items():
            print(f"Fitting {estimator_name}...", end="")
            t0 = perf_counter()
            estimator.fit(X, y)
            records.append({
                "n": n,
                "time": perf_counter() - t0,
                "estimator": estimator_name,
            })
            print(f"Finished in {records[-1]['time']:.2f} s.")
            (
                pd.DataFrame
                .from_records(records)
                .to_csv(outpath, index=False)
            )


if __name__ == "__main__":
    main()
