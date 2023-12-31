"""Make the interaction dataset from the BindingDB database.

This script takes the files generated by `parse_binding_db.py` 
and creates:

1. mirna-mirna similarity matrix;
2. Protein-protein similarity matrix;
3. mirna-protein interaction matrix with values from the metric specified to
`parse_binding_db.py`;
4. Binary mirna-protein interaction matrix, based on the `--threshold`
specified.

More information available in the --help message.
"""
# Author: Pedro Ilidio <pedrilidio@gmail.com>, 2023
# License: BSD 3 clause
import warnings
from argparse import ArgumentParser
from typing import Callable, Sequence
from pathlib import Path
from itertools import combinations_with_replacement
import pandas as pd
import numpy as np
from Bio.Align import substitution_matrices, PairwiseAligner
from joblib import Parallel, delayed


def normalize_similarity_matrix(similarity_matrix: np.ndarray):
    """Normalize Smith-Waterman scores.

    Normalize values of a similarity matrix composed of alignment scores to the
    0-1 interval.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Square similarity matrix between sequences.
    """
    diag = np.diag(similarity_matrix.values)
    denom = np.sqrt(diag[:, None] * diag[None, :])
    return similarity_matrix / denom


def compute_similarity_matrix(
    seqs: pd.Series | Sequence,
    similarity_func: Callable,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute a similarity matrix for the given sequences.

    Parameters
    ----------
    seqs : pd.Series | Sequence[str]
        A series or list-like of biological sequences.
    aligner : PairwiseAligner
        A Biopython pairwise aligner.
    n_jobs : int, optional
        Number of parallel jobs to run, by default 1.

    Returns
    -------
    pd.DataFrame
        A similarity matrix.
    """
    seqs = pd.Series(seqs)

    def compute_similarity(i, j):
        return i, j, similarity_func(seqs[i], seqs[j])

    total: int = (len(seqs) * (len(seqs) + 1)) // 2
    batch_size = int(np.ceil(total / n_jobs))

    print(f"Computing {total} pairwise similarities...")
    records = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=11)(
        delayed(compute_similarity)(i, j)
        for i, j in combinations_with_replacement(seqs.index, 2)
    )

    print("Building similarity matrix...", end=" ")
    similarity_matrix = pd.DataFrame.from_records(records, columns=["i", "j", "score"])
    similarity_matrix = similarity_matrix.pivot(index="i", columns="j", values="score")
    # Fill in the lower triangle
    similarity_matrix = similarity_matrix.combine_first(similarity_matrix.T)
    # Fix possible ordering change due to parallelization
    similarity_matrix = similarity_matrix.loc[seqs.index, seqs.index]
    print("Done.")

    return similarity_matrix


def compute_nucleotide_similarity_matrix(seqs, **kwargs):
    """Calculate the Smith-Waterman score between nucleotide sequences.

    Parameters
    ----------
    seqs : pd.Series | Sequence[str]
        A series or list-like of protein sequences.
    n_jobs : int, optional
        Number of parallel jobs to run, by default 1.

    Returns
    -------
    pd.DataFrame
        Square similarity matrix with pairwise Smith-Waterman alignment scores
        between the sequences.
    """
    aligner = PairwiseAligner(
        mode="global",
        substitution_matrix=substitution_matrices.load("BLASTN"),
        open_gap_score=0,  # TODO: check if this is correct
        extend_gap_score=0,
    )
    # Remove sequences with non-standard amino acids/nucleotides
    mask = seqs.apply(lambda s: all(l in aligner.alphabet for l in s))
    if not mask.all():
        warnings.warn(
            f"Found {len(seqs) - mask.sum()} sequences with non-standard "
            "amino acids/nucleotides. Dropping them."
        )
    seqs = seqs[mask]
    if seqs.empty:
        raise ValueError("Sequences seem to use a different alphabet than the aligner.")
    return compute_similarity_matrix(seqs, aligner.score, **kwargs)


def drop_single_interactions(
    interactions,
    row_sample_ids: str,  # column name of the row sample IDs (mirnas)
    col_sample_ids: str,  # column name of the column sample IDs (targets)
    max_iter=15,
    min_rows=2,
    min_cols=2,
) -> pd.DataFrame:
    """Drop single interactions from the interaction table.

    Here, "rows" refer to samples referring to the rows of the final interaction
    matrix (mirnas) and "columns" refer to samples referring to the columns of
    the final interaction matrix (targets).

    Parameters
    ----------
    interactions : pd.DataFrame
        Interaction table.
    row_sample_ids : str
        Name of the column in interactions containing the row IDs (mirnas).
    col_sample_ids : str
        Name of the column in interactions containing the column IDs (targets).
    max_iter : int, optional
        Maximum number of iterations, by default 15.
    min_rows : int, optional
        Minimum number of interactions for each row sample, by default 2.
    min_cols : int, optional
        Minimum number of interactions for each column sample, by default 2.

    Returns
    -------
    pd.DataFrame
        Interaction table with single interactions dropped.
    """
    n_col_samples = len(interactions[col_sample_ids].unique())
    n_row_samples = len(interactions[row_sample_ids].unique())

    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")

    for _ in range(max_iter):
        mirna_counts = interactions[row_sample_ids].value_counts()
        interactions = interactions[
            interactions[row_sample_ids].isin(
                mirna_counts[mirna_counts >= min_rows].index
            )
        ]
        target_counts = interactions[col_sample_ids].value_counts()
        interactions = interactions[
            interactions[col_sample_ids].isin(
                target_counts[target_counts >= min_cols].index
            )
        ]
        n_remaining_cols = len(interactions[col_sample_ids].unique())
        n_remaining_rows = len(interactions[row_sample_ids].unique())
        print(f"Dropped {n_row_samples - n_remaining_rows} row samples.")
        print(f"Dropped {n_col_samples - n_remaining_cols} column samples.")

        if n_remaining_cols == n_col_samples and n_remaining_rows == n_row_samples:
            break
        n_col_samples = n_remaining_cols
        n_row_samples = n_remaining_rows

    print(f"Remaining {n_remaining_rows} row samples.")
    print(f"Remaining {n_remaining_cols} column samples.")
    return interactions


def make_interaction_dataset(
    interactions_table_input: Path,
    final_interactions_output: Path,
    mirna_sequences_input: Path,
    target_sequences_input: Path,
    mirna_output: Path,
    target_output: Path,
    mirna_similarity_output: Path,
    target_similarity_output: Path,
    norm_mirna_output: Path,
    norm_target_output: Path,
    interaction_matrix_output: Path,
    n_jobs: int = 1,
):
    # Read interactions table:
    # mirna_id       target_gene_id
    # hsa-miR-1-3p   1003
    # hsa-miR-1-3p   134
    # hsa-miR-1-3p   2412

    interactions = pd.read_table(interactions_table_input)
    print(f"Loaded {len(interactions)} interactions.")

    # Drop duplicate interactions
    print(f"Found {interactions.duplicated().sum()} duplicate interactions.")
    interactions = interactions.drop_duplicates()

    # Iteratively filter single interactions
    # TODO: These numbers are arbitrary, we should provide them as parameters or define
    # a heuristic.
    interactions = drop_single_interactions(
        interactions,
        row_sample_ids="mirna_id",
        col_sample_ids="target_gene_id",
        min_rows=10,
        min_cols=100,
    )
    # Read mirna sequences:
    # mirna_id       sequence
    # hsa-miR-1-3p   UGGAAUGUAAAGAAGUAUGG
    # hsa-miR-1-5p   UGGAAUGUAAAGAAGUAUGGA
    # hsa-miR-100-3p UAAUACUGUCUGGUAAUGAUGA
    mirna_seqs = pd.read_table(
        mirna_sequences_input,
        index_col=0,
    ).sequence.str.replace("U", "T")

    # Select only the mirnas that still are in the interaction table
    mirna_seqs = mirna_seqs[mirna_seqs.index.isin(interactions.mirna_id.unique())]
    mirna_output.parent.mkdir(exist_ok=True, parents=True)
    mirna_seqs.to_csv(mirna_output, sep="\t")
    print(f"{len(mirna_seqs)} miRNA sequences saved to {mirna_output}.")

    # Read target gene sequences:
    # gene_id  sequence
    # 1003     ATGGCGAGATGATGACGAGTTCG...
    # 134      TGGCGAGATGATGACGAGTTCG...
    # 2412     CGAGATGATGACGAGTTCG...
    target_seqs = pd.read_table(
        target_sequences_input,
        index_col=0,
    ).sequence

    # Select only the targets that still are in the interactions table
    target_seqs = target_seqs[
        target_seqs.index.isin(interactions.target_gene_id.unique())
    ]
    target_output.parent.mkdir(exist_ok=True, parents=True)
    target_seqs.to_csv(target_output, sep="\t")
    print(f"{len(target_seqs)} target gene sequences saved to {target_output}.")

    # Filter interactions table to only contain miRNAs and targets that are
    # still in the sequences tables.
    interactions = interactions[
        interactions.mirna_id.isin(mirna_seqs.index)
        & interactions.target_gene_id.isin(target_seqs.index)
    ]
    final_interactions_output.parent.mkdir(exist_ok=True, parents=True)
    interactions.to_csv(final_interactions_output, sep="\t", index=False)
    print(
        f"{len(interactions)} remaining interactions saved to "
        f"{final_interactions_output}."
    )

    # Build interaction matrix
    print("Building interaction matrix...")
    interaction_matrix = pd.crosstab(
        interactions.mirna_id,
        interactions.target_gene_id,
    ).loc[mirna_seqs.index, target_seqs.index]

    interaction_matrix_output.parent.mkdir(exist_ok=True, parents=True)
    interaction_matrix.to_csv(interaction_matrix_output, sep="\t")
    print(f"Interaction matrix saved to {interaction_matrix_output}.")

    # Compute similarity matrices
    print("Calculating mirna-mirna similarities...")
    mirna_similarity_matrix = compute_nucleotide_similarity_matrix(
        mirna_seqs,
        n_jobs=n_jobs,
    )
    mirna_similarity_output.parent.mkdir(exist_ok=True, parents=True)
    mirna_similarity_matrix.to_csv(mirna_similarity_output, sep="\t")
    print(f"miRNA similarity matrix saved to {mirna_similarity_output}.")

    norm_mirna_similarity_matrix = normalize_similarity_matrix(mirna_similarity_matrix)
    norm_mirna_output.parent.mkdir(exist_ok=True, parents=True)
    norm_mirna_similarity_matrix.to_csv(norm_mirna_output, sep="\t")
    print(f"Normalized miRNA similarity matrix saved to {norm_mirna_output}.")

    print("Calculating target-target similarities...")
    target_similarity_matrix = compute_nucleotide_similarity_matrix(
        target_seqs,
        n_jobs=n_jobs,
    )
    target_similarity_output.parent.mkdir(exist_ok=True, parents=True)
    target_similarity_matrix.to_csv(target_similarity_output, sep="\t")
    print(f"Target protein similarity matrix saved to {target_similarity_output}.")

    norm_target_similarity_matrix = normalize_similarity_matrix(
        target_similarity_matrix
    )
    norm_target_output.parent.mkdir(exist_ok=True, parents=True)
    norm_target_similarity_matrix.to_csv(norm_target_output, sep="\t")
    print(f"Normalized target protein similarity matrix saved to {norm_target_output}.")


def main():
    interaction_table_input = Path("interactions.tsv")
    mirna_sequences_input = Path("mirna_sequences.tsv")
    target_sequences_input = Path("gene_sequences.tsv")
    mirna_output = Path("final/final_mirna_seqs.tsv")
    target_output = Path("final/final_target_seqs.tsv")
    mirna_similarity_output = Path("final/mirna_similarity.tsv")
    target_similarity_output = Path("final/target_similarity.tsv")
    norm_mirna_output = Path("final/normalized_mirna_similarity.tsv")
    norm_target_output = Path("final/normalized_target_similarity.tsv")
    interaction_matrix_output = Path("final/interaction_matrix.tsv")
    final_interactions_output = Path("final/final_interactions.tsv")

    parser = ArgumentParser(
        description=(
            "Format miRNA-target interaction data as a bipartite edge-prediction task."
            " This script will compute similarity matrices for the"
            " miRNA and target genes using "
            " Smith-Waterman scores."
            " The alignment scores will be normalized so that the similarity matrices"
            " have values between 0 and 1."
        )
    )

    parser.add_argument(
        "--interactions_table_input",
        type=Path,
        default=interaction_table_input,
        help="Path to the interactions table.",
    )
    parser.add_argument(
        "--mirna_sequences_input",
        type=Path,
        default=mirna_sequences_input,
        help="Path to the tab-separated values file relating miRNA IDs to sequences.",
    )
    parser.add_argument(
        "--target_sequences_input",
        type=Path,
        default=target_sequences_input,
        help="Path to the tab-separated values file relating gene IDs to sequences.",
    )
    parser.add_argument(
        "--final_interactions_output",
        type=Path,
        default=final_interactions_output,
        help="Path to save the remaining interactions table.",
    )
    parser.add_argument(
        "--mirna_output",
        type=Path,
        default=mirna_output,
        help="Path to save the mirna sequences.",
    )
    parser.add_argument(
        "--target_output",
        type=Path,
        default=target_output,
        help="Path to save the target protein sequences.",
    )
    parser.add_argument(
        "--mirna_similarity_output",
        type=Path,
        default=mirna_similarity_output,
        help="Path to save the miRNA similarity matrix.",
    )
    parser.add_argument(
        "--target_similarity_output",
        type=Path,
        default=target_similarity_output,
        help="Path to save the target protein similarity matrix.",
    )
    parser.add_argument(
        "--norm_mirna_output",
        type=Path,
        default=norm_mirna_output,
        help="Path to save the normalized miRNA similarity matrix.",
    )
    parser.add_argument(
        "--norm_target_output",
        type=Path,
        default=norm_target_output,
        help="Path to save the normalized target protein similarity matrix.",
    )
    parser.add_argument(
        "--interaction_matrix_output",
        type=Path,
        default=interaction_matrix_output,
        help="Path to save the interaction matrix.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs to use for similarity calculation.",
    )
    args = parser.parse_args()

    make_interaction_dataset(
        interactions_table_input=args.interactions_table_input,
        mirna_sequences_input=args.mirna_sequences_input,
        target_sequences_input=args.target_sequences_input,
        final_interactions_output=args.final_interactions_output,
        mirna_output=args.mirna_output,
        target_output=args.target_output,
        mirna_similarity_output=args.mirna_similarity_output,
        target_similarity_output=args.target_similarity_output,
        norm_mirna_output=args.norm_mirna_output,
        norm_target_output=args.norm_target_output,
        interaction_matrix_output=args.interaction_matrix_output,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
