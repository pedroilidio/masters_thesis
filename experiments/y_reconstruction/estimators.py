import sys
from copy import deepcopy
import numpy as np

from scipy.stats import loguniform
from sklearn.base import MetaEstimatorMixin, ClassifierMixin
from sklearn.metrics import average_precision_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import (
    MultiOutputRegressor,
    MultiOutputClassifier,
)
from sklearn.svm import SVR, SVC

from bipartite_learn.pipeline import make_multipartite_pipeline
from bipartite_learn.preprocessing.multipartite import DTHybridSampler
from bipartite_learn.preprocessing.monopartite import (
    TargetKernelLinearCombiner,
    TargetKernelDiffuser,
    SimilarityDistanceSwitcher,
    SymmetryEnforcer,
)
from bipartite_learn.neighbors import WeightedNeighborsRegressor
from bipartite_learn.wrappers import LocalMultiOutputWrapper
from bipartite_learn.matrix_factorization import (
    NRLMFSampler,
    NRLMFClassifier,
    DNILMFSampler,
    DNILMFClassifier,
)
from bipartite_learn.model_selection import (
    MultipartiteGridSearchCV,
    MultipartiteRandomizedSearchCV,
    make_multipartite_kfold,
)

# sys.path.insert(0, "..")
import wrappers


kfold_5_shuffle_diag = make_multipartite_kfold(
    n_parts=2,  # Bipartite
    cv=5,
    shuffle=True,
    diagonal=True,
    random_state=0,
)


blmnii_rls = MultipartiteGridSearchCV(
    make_multipartite_pipeline(
        SymmetryEnforcer(),
        TargetKernelLinearCombiner(),
        LocalMultiOutputWrapper(
            primary_rows_estimator=WeightedNeighborsRegressor(
                metric="precomputed",
                weights="similarity",
            ),
            primary_cols_estimator=WeightedNeighborsRegressor(
                metric="precomputed",
                weights="similarity",
            ),
            secondary_rows_estimator=KernelRidge(kernel="precomputed"),
            secondary_cols_estimator=KernelRidge(kernel="precomputed"),
            independent_labels=False,
        ),
    ),
    param_grid={
        "targetkernellinearcombiner__samplers__alpha": [
            0.0,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            1.0,
        ],
    },
    cv=kfold_5_shuffle_diag,
    n_jobs=3,
    scoring="neg_mean_squared_error",
    pairwise=True,
)

blmnii_svm = MultipartiteGridSearchCV(
    make_multipartite_pipeline(
        SymmetryEnforcer(),
        TargetKernelLinearCombiner(),
        LocalMultiOutputWrapper(
            primary_rows_estimator=WeightedNeighborsRegressor(
                metric="precomputed",
                weights="similarity",
            ),
            primary_cols_estimator=WeightedNeighborsRegressor(
                metric="precomputed",
                weights="similarity",
            ),
            secondary_rows_estimator=MultiOutputRegressor(SVR(kernel="precomputed")),
            secondary_cols_estimator=MultiOutputRegressor(SVR(kernel="precomputed")),
            independent_labels=True,
        ),
    ),
    param_grid={
        "targetkernellinearcombiner__samplers__alpha": [
            0.0,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            1.0,
        ],
    },
    cv=kfold_5_shuffle_diag,
    n_jobs=3,
    scoring="neg_mean_squared_error",
    pairwise=True,
)

dthybrid_regressor = make_multipartite_pipeline(
    SymmetryEnforcer(),
    DTHybridSampler(),
    LocalMultiOutputWrapper(
        primary_rows_estimator=WeightedNeighborsRegressor(
            metric="precomputed",
            weights="similarity",
        ),
        primary_cols_estimator=WeightedNeighborsRegressor(
            metric="precomputed",
            weights="similarity",
        ),
        secondary_rows_estimator=WeightedNeighborsRegressor(
            metric="precomputed",
            weights="similarity",
        ),
        secondary_cols_estimator=WeightedNeighborsRegressor(
            metric="precomputed",
            weights="similarity",
        ),
        independent_labels=True,
    ),
)

# van Laarhoven
lmorls = MultipartiteGridSearchCV(
    make_multipartite_pipeline(
        TargetKernelLinearCombiner(),
        LocalMultiOutputWrapper(
            primary_rows_estimator=KernelRidge(kernel="precomputed"),
            primary_cols_estimator=KernelRidge(kernel="precomputed"),
            secondary_rows_estimator=KernelRidge(kernel="precomputed"),
            secondary_cols_estimator=KernelRidge(kernel="precomputed"),
            independent_labels=False,
        ),
    ),
    param_grid={
        "targetkernellinearcombiner__samplers__alpha": [
            0.0,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            1.0,
        ],
    },
    cv=kfold_5_shuffle_diag,
    n_jobs=3,
    scoring="neg_mean_squared_error",
    pairwise=True,
)

# The original proposal cannot be used:
# common_param_options = [2**-2, 2**-1, 1, 2]  # 18_432 parameter combinations!
common_param_options = loguniform(2**-2, 2)

nrlmf_grid = MultipartiteRandomizedSearchCV(
    NRLMFClassifier(),
    param_distributions=dict(
        lambda_rows=common_param_options,
        lambda_cols=common_param_options,
        alpha_rows=common_param_options,
        alpha_cols=common_param_options,
        learning_rate=common_param_options,
        n_neighbors=[3, 5, 10],
        n_components_rows=[50, 100],
        # n_components_cols="same",
    ),
    scoring="average_precision",
    cv=deepcopy(kfold_5_shuffle_diag),
    refit=True,
    verbose=1,
    n_jobs=3,
    n_iter=100,
    random_state=0,
    pairwise=True,
)

nrlmf = nrlmf_grid


dnilmf_grid = MultipartiteRandomizedSearchCV(
    DNILMFClassifier(),
    param_distributions=dict(
        lambda_rows=common_param_options,
        lambda_cols=common_param_options,
        beta=[0.1, 0.2, 0.4, 0.5],
        gamma=[0.1, 0.2, 0.4, 0.5],
        learning_rate=common_param_options,
        n_neighbors=[3, 5, 10],
        n_components_rows=[50, 100],
    ),
    scoring="average_precision",
    cv=deepcopy(kfold_5_shuffle_diag),
    refit=True,
    verbose=1,
    n_jobs=3,
    n_iter=100,
    random_state=0,
    pairwise=True,
)

dnilmf = make_multipartite_pipeline(
    SymmetryEnforcer(),
    TargetKernelDiffuser(),
    dnilmf_grid,
    memory="/tmp",
)


def nrlmf_y_reconstruction_wrapper(estimator):
    return wrappers.RegressorToBinaryClassifier(
        make_multipartite_pipeline(
            SymmetryEnforcer(),
            wrappers.ClassifierAsSampler(nrlmf_grid, keep_positives=True),
            estimator,
            memory="/tmp",
        )
    )


def dnilmf_y_reconstruction_wrapper(estimator):
    return wrappers.RegressorToBinaryClassifier(
        make_multipartite_pipeline(
            SymmetryEnforcer(),
            TargetKernelDiffuser(),
            wrappers.ClassifierAsSampler(dnilmf_grid, keep_positives=True),
            estimator,
            memory="/tmp",
        )
    )
