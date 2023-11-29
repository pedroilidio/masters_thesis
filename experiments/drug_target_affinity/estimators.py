import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
import bipartite_learn.ensemble
from bipartite_learn.base import BaseBipartiteEstimator
from bipartite_learn.preprocessing.monopartite import SymmetryEnforcer
from bipartite_learn.pipeline import make_multipartite_pipeline

import DeepPurpose.DTI
import DeepPurpose.utils

from drug_target_affinity.deep_purpose_wrapper import DeepPurposeWrapper


class KnTransformer(BaseEstimator, TransformerMixin):
    """Transforms target Kn to p (using log10)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return DeepPurpose.utils.convert_y_unit(X, "nM", "p")

    def inverse_transform(self, X):
        return DeepPurpose.utils.convert_y_unit(X, "p", "nM")


class BipartiteTransformedTargetRegressor(
    TransformedTargetRegressor,
    BaseBipartiteEstimator,
):
    pass


def wrap_forest(forest):
    """A common wrapper for forest estimators"""
    return make_multipartite_pipeline(
        SymmetryEnforcer(),
        BipartiteTransformedTargetRegressor(
            regressor=forest,
            transformer=KnTransformer(),
        ),
    )


print(f"* {torch.cuda.is_available()=}")
print(f"* {torch.cuda.device_count()=}")
print(f"* {torch.cuda.get_device_name(0)=}")

# NOTE: DeepPurpose selects GPU automatically if available.
deep_dta = DeepPurposeWrapper(
    DeepPurpose.utils.generate_config(
        drug_encoding="CNN",
        target_encoding="CNN",
        cls_hidden_dims=[1024, 1024, 512],
        train_epoch=100,
        LR=0.001,
        batch_size=256,
        cnn_drug_filters=[32, 64, 96],
        cnn_target_filters=[32, 64, 96],
        cnn_drug_kernels=[4, 6, 8],
        cnn_target_kernels=[4, 8, 12],
        cuda_id=0,
    ),
    # Selected balanced number of unknown interactions:
    under_sampler=RandomUnderSampler(random_state=0),
    binarizer=lambda y: (y > y.min()).astype(int),
)

### MolTrans config
# Source: https://github.com/kexinhuang12345/MolTrans/blob/master/config.py
#
#    config['batch_size'] = 16
#    config['input_dim_drug'] = 23532
#    config['input_dim_target'] = 16693
#    config['train_epoch'] = 13
#    config['max_drug_seq'] = 50
#    config['max_protein_seq'] = 545
#    config['emb_size'] = 384
#    config['dropout_rate'] = 0.1
#
#    #DenseNet
#    config['scale_down_ratio'] = 0.25
#    config['growth_rate'] = 20
#    config['transition_rate'] = 0.5
#    config['num_dense_blocks'] = 4
#    config['kernal_dense_size'] = 3
#
#    # Encoder
#    config['intermediate_size'] = 1536
#    config['num_attention_heads'] = 12
#    config['attention_probs_dropout_prob'] = 0.1
#    config['hidden_dropout_prob'] = 0.1
#    config['flat_dim'] = 78192

moltrans = DeepPurposeWrapper(
    DeepPurpose.utils.generate_config(
        drug_encoding="Transformer",
        target_encoding="Transformer",
        input_dim_drug=23532,
        input_dim_protein=16693,
        train_epoch=13,
        batch_size=16,
        transformer_dropout_rate=0.1,
        transformer_emb_size_drug=384,
        transformer_intermediate_size_drug=1536,
        transformer_num_attention_heads_drug=12,
        transformer_emb_size_target=384,
        transformer_intermediate_size_target=1536,
        transformer_num_attention_heads_target=12,
        transformer_attention_probs_dropout=0.1,
        transformer_hidden_dropout_rate=0.1,
        cls_hidden_dims=[1024, 1024, 512],
        LR=0.001,
    ),
    # Selected balanced number of unknown interactions:
    under_sampler=RandomUnderSampler(random_state=0),
    binarizer=lambda y: (y > y.min()).astype(int),
)

brf_gmosa = bipartite_learn.ensemble.BipartiteRandomForestRegressor(
    bipartite_adapter="gmosa",
    criterion="squared_error",
    max_row_features="sqrt",
    max_col_features="sqrt",
    n_estimators=1000,
    n_jobs=3,
)

brf_gso = bipartite_learn.ensemble.BipartiteRandomForestRegressor(
    bipartite_adapter="gmosa",
    criterion="squared_error_gso",
    max_row_features="sqrt",
    max_col_features="sqrt",
    n_estimators=1000,
    n_jobs=3,
)

bxt_gmosa = bipartite_learn.ensemble.BipartiteExtraTreesRegressor(
    bipartite_adapter="gmosa",
    criterion="squared_error",
    n_estimators=1000,
    n_jobs=3,
)

bxt_gso = bipartite_learn.ensemble.BipartiteExtraTreesRegressor(
    bipartite_adapter="gmosa",
    criterion="squared_error_gso",
    n_estimators=1000,
    n_jobs=3,
)

bgbm = bipartite_learn.ensemble.BipartiteGradientBoostingRegressor(
    bipartite_adapter="gmosa",
    criterion="friedman_gso",
    n_estimators=1000,
)

