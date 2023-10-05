from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

outdir = Path("figures")
data_sizes = {
    "enzymes": (445, 664),
    "ion_channels": (210, 204),
    "gpcr": (223, 95),
    "nuclear_receptors": (54, 26),
}


def main(outdir=outdir):
    data = pd.read_table("results.tsv", header=[0, 1])
    data.columns = map(lambda x: "_".join(x), data.columns)
    data = data[~data.dataset_name.isin(("nuclear_receptors", "gpcr"))]

    data["n_samples"] = data.dataset_name.apply(
        lambda x: data_sizes[x][0] * data_sizes[x][1]
    )
    data["log_n_samples"] = np.log2(data.n_samples)
    data["log_fit_time"] = np.log2(data.results_fit_time)
    data["log_score_time"] = np.log2(data.results_score_time)

    sns.lmplot(
        data=data,
        y="log_fit_time",
        x="log_n_samples",
        hue="estimator_name",
        #x_jitter=0.1,
        markers=".",
    )

    outdir.mkdir(exist_ok=True, parents=True)
    plt.savefig(outdir / "time_vs_data_size.png")


if __name__ == "__main__":
    main()
