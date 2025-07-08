#!/usr/bin/env python3
"""
Aggregate row counts per Subject using Dask.

Configuration
-------------
Edit the **USER PARAMETERS** section below to point to your own files.
After that, simply run:

    python dask_row_counts.py

No command‑line flags are necessary anymore.
"""

import dask.dataframe as dd
from dask.distributed import Client
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns
import random

# === USER PARAMETERS =========================================================
ROWS_PATH = Path("/REDACTED/PATH")           # CSV or Parquet containing the rows table
#ROWS_PATH = Path("/REDACTED/PATH")
METADATA_PATH = Path("/REDACTED/PATH")  # CSV mapping File_ID -> Subject, Author
OUTPUT_PLOT = Path("rows_per_Subject.pdf")   # Where to save the bar‑plot PNG
N_WORKERS = None                                # Set to None to disable the local cluster
CACHED_COUNTS = Path("counts_with_meta.csv")     # Cached intermediate CSV
#CACHED_COUNTS = Path("counts_with_meta_before_sampling.csv")
# =============================================================================


def get_row_counts(rows_path: Path) -> dd.DataFrame:
    """Return a Dask Series of row counts per File_ID (lazy)."""
    if rows_path.suffix == ".parquet":
        ddf = dd.read_parquet(rows_path, columns=["File_ID"])
    else:
        ddf = dd.read_csv(
            rows_path,
            usecols=["File_ID"],  # read only what we need
            assume_missing=True,
            blocksize="64MB",
        )
    # Rename File_ID to File_index
    ddf = ddf.rename(columns={"File_ID": "File_index"})
    return ddf.groupby("File_index").size().rename("row_count")


def build_metadata(metadata_path: Path) -> dd.DataFrame:
    """Return a Dask DataFrame with File_ID, Individual, Publication columns (lazy)."""
    cols = ["File_index", "Subject", "Author"]
    meta_ddf = (
        dd.read_csv(
            metadata_path,
            usecols=cols,
            assume_missing=True,
            blocksize="16MB",
        )
        .dropna(subset=["File_index", "Subject"])
        .reset_index(drop=True)
    )
    return meta_ddf


def create_mapping(meta_ddf: dd.DataFrame) -> dict:
    """Build {file_id: (individual, publication)} mapping dict (in‑memory)."""
    return (
        meta_ddf.compute()
        .set_index("File_index")
        .apply(lambda r: (r["Subject"], r["Author"]), axis=1)
        .to_dict()
    )


def plot_counts(ind_counts, output_plot: Path):
    """Create and save a bar plot of row counts per individual."""
    plt.figure(figsize=(12, 6))

    n = len(ind_counts)
    cmap = plt.cm.get_cmap("YlGnBu", n * 2)  # oversample for smoother variety

    # Build a colour list that alternates from darker to lighter shades
    base = np.linspace(0.15, 0.85, n)        # skip extreme ends of the scale
    colours = [cmap(base[i]) for i in range(n)]

    # Swap every second colour to create an alternating effect
    for i in range(1, n, 2):
        colours[i] = cmap(base[i] + 0.1) if base[i] + 0.1 <= 1 else cmap(base[i] - 0.1)

    ax = ind_counts.plot(kind="bar",
                         width=0.95,
                         color=colours,
                         edgecolor="none",
                         linewidth=0)
    
    # Set grid below the bars and enable light gray horizontal lines
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.3)

    # Remove top and right spines (box lines)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



    plt.ylabel("Number of sequences")
    plt.title("Sequences per individual")
    # remove the x-axis labels
    plt.gca().set_xticklabels([])
    # make the x-axis labels thinner
    plt.gca().tick_params(axis='x', width=0.3)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, backend="cairo")
    print(f"Plot saved to {output_plot}")

def plot_counts2(ind_counts, output_plot: Path):
    """Create and save a bar plot of row counts per individual.

    Colours: 5 fixed shades from the YlGnBu palette, cycling as needed.
    """
    plt.figure(figsize=(12, 6))

    n = len(ind_counts)

    # --- SIMPLE CYCLING COLOUR SCHEME --------------------------------------
    cmap = plt.cm.get_cmap("YlGnBu", 10)          # exactly 10 discrete shades
    base_palette = [cmap(i) for i in range(10)]   # list of RGBA tuples
    colours = [base_palette[i % 10] for i in range(n)]
    # -----------------------------------------------------------------------

    ax = ind_counts.plot(kind="bar",
                         width=0.95,
                         color=colours,
                         edgecolor="none",
                         linewidth=0)

    # Grid below the bars
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.3)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.ylabel("Number of sequences")
    plt.title("Sequences per individual")

    # Hide x-axis tick labels (they tend to overlap if many individuals)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", width=0.3)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, backend="cairo")
    print(f"Plot saved to {output_plot}")

def plot_counts3(ind_counts, output_plot: Path, *, seed: int | None = None):
    """Bar plot of row counts per individual with random YlGnBu colours."""
    if seed is not None:                # reproducibility toggle
        random.seed(seed)               # or npr.seed(seed)

    plt.figure(figsize=(8, 6))
    n = len(ind_counts)

    # --- 5-colour palette ---------------------------------------------------
    cmap = plt.cm.get_cmap("YlGnBu", 100)
    palette = [cmap(i) for i in range(100)]
    colours = [random.choice(palette) for _ in range(n)]  # NEW: random pick
    # -----------------------------------------------------------------------

    # modify ind_counts to only include subjects with more than 100 sequences
    ind_counts = ind_counts[ind_counts > 1000]

    ax = ind_counts.plot(kind="bar",
                         width=0.95,
                         color=colours,
                         edgecolor="none",
                         linewidth=0)

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.ylabel("Number of sequences")
    plt.title("Sequences per individual")
    ax.set_xticklabels([])
    ax.tick_params(axis="x", width=0.3)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, backend="cairo")
    print(f"Plot saved to {output_plot}")


def main():
    # Load metadata once (lightweight) so we can always build the file‑ID mapping
    meta_ddf = build_metadata(METADATA_PATH)

    if CACHED_COUNTS.exists():
        # Fast path: reuse cached counts_with_meta CSV
        print(f"Found cached counts: {CACHED_COUNTS}. Using it to regenerate plot …")
        counts_with_meta = dd.read_csv(CACHED_COUNTS, assume_missing=True).compute()
    else:
        # Slow path: compute counts and merge with metadata
        print("No cached counts found. Computing row totals from raw data …")
        row_counts_ddf = get_row_counts(ROWS_PATH)

        counts_with_meta = (
            row_counts_ddf.reset_index()
            .merge(meta_ddf, on="File_index", how="left")
            .compute()
        )

        # Persist for next run
        counts_with_meta.to_csv(CACHED_COUNTS, index=False)
        print(f"Cached counts saved to {CACHED_COUNTS}")

    # Aggregate rows per individual
    ind_counts = (
        counts_with_meta.groupby("Subject")["row_count"]
        .sum()
        .sort_values(ascending=False)
    )

    # Mapping (may be useful downstream)
    file_mapping = create_mapping(meta_ddf)

    # Plot
    plot_counts3(ind_counts, OUTPUT_PLOT)

    # Console summary
    print(f"Built mapping for {len(file_mapping):,} file IDs. Example entries:")
    for k, v in list(file_mapping.items())[:5]:
        print(f"  {k}: {v}")

    print("Done.")


if __name__ == "__main__":
    main()