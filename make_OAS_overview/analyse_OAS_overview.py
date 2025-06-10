import dask.dataframe as dd
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from oas_analysis_utils import (
    load_oas_overview,
    plot_grouped_data,
    plot_grouped_data_two_bars,
    plot_filtered_labels,
)


INPUT_FILE = Path("assets/OAS_overview.csv")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)


# Load and clean the overview data with Dask
print("Reading overview CSV...")
ddf = load_oas_overview(INPUT_FILE)
print("File read complete")

df = ddf.compute()

"""
# Weighted histogram of sequences per age
plt.figure(figsize=(10, 6))
plt.hist(
    df["Age"],
    bins=20,
    weights=df["Unique_sequences"],
    edgecolor="black",
    alpha=0.7,
)
plt.xlabel("Age")
plt.ylabel("Number of Unique Sequences")
plt.title("Histogram of Unique Sequences per Age")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig(PLOT_DIR / "sequences_per_age.pdf", format="pdf")
print("Histogram saved as sequences_per_age.pdf")
"""

# Publications with subject "no"
print(ddf[ddf["Subject"] == "no"]["Author"].unique().compute())

# Individuals present in multiple publications
individual_counts = ddf.groupby("Subject")["Author"].nunique().compute()
multiple_publications = individual_counts[individual_counts > 1].index
for individual in multiple_publications:
    publications = df[df["Subject"] == individual]["Author"].unique()
    print(f"Individual: {individual}, Publications: {list(publications)}")

# Unique publications
unique_publications = ddf["Author"].unique().compute()
print(f"Unique publications: {list(unique_publications)}")

# Total size calculations
size_series = dd.to_numeric(ddf["Size_MB"], errors="coerce").fillna(0)
total_size = size_series.sum().compute()
print(f"Total size in MB: {total_size}")
print(f"Total size in GB: {total_size / 1024}")
print(f"Total size in TB: {total_size / 1024 / 1024}")

# Grouped data per author
grouped_data = ddf.groupby("Author")["Unique_sequences"].sum().compute()
print(grouped_data)
(grouped_data).to_csv("grouped_data.csv")

# Plotting
plot_grouped_data(ddf, ["Author"], "Size_MB", "Total Size (MB)", "Publication", "Total Size by Publication", PLOT_DIR / "total_size_by_publication.pdf")
plot_grouped_data(ddf, ["Author"], "Total_sequences", "Total sequences", "Publication", "Total sequences by Publication", PLOT_DIR / "total_sequences_by_publication.pdf")
plot_grouped_data(ddf, ["Author"], "Unique_sequences", "Unique sequences", "Publication", "Unique sequences by Publication", PLOT_DIR / "unique_sequences_by_publication.pdf")
plot_grouped_data(ddf, ["Subject", "Author"], "Unique_sequences", "Unique sequences", "Publication", "Unique Sequences by Subject and Publication", PLOT_DIR / "unique_sequences_by_subject_log.pdf", log_scale=True)
plot_grouped_data(ddf, ["Subject", "Author"], "Unique_sequences", "Unique sequences", "Publication", "Unique Sequences by Subject and Publication", PLOT_DIR / "unique_sequences_by_subject.pdf")
plot_grouped_data(ddf, ["Species", "Chain"], "Unique_sequences", "Unique sequences", "Species and Chain" ,"Unique Sequences by Species and Chain", PLOT_DIR / "unique_sequences_by_species_chain.pdf")
plot_grouped_data_two_bars(
    ddf,
    ["Species", "Chain"],
    "Unique_sequences",
    "Unique sequences",
    "Species",
    "Unique Sequences by Species and Chain",
    PLOT_DIR / "unique_sequences_by_species_chain_grouped.pdf",
)


# Histogram for subjects vs unique sequences per individual
subject_sizes = ddf.groupby(["Subject", "Author"])["Unique_sequences"].sum().compute().reset_index()

bin_edges = [0, 100, 1000, 10000, 100000, 1000000, 10000000, float("inf")]
subject_sizes_bins = pd.cut(subject_sizes["Unique_sequences"], bins=bin_edges)

bin_counts = subject_sizes_bins.value_counts().sort_index()
plt.figure(figsize=(10, 6))
bin_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.xlabel("Range of Unique Sequences per Individual", fontsize=12)
plt.ylabel("Number of Subjects", fontsize=12)
plt.title("Histogram of Subjects by Range of Unique Sequences", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=10)
plt.tight_layout()
plt.savefig(PLOT_DIR / "histogram_unique_sequences_by_subject.pdf")

# Unique sequences per species
species_sizes = ddf.groupby("Species")["Unique_sequences"].sum().compute().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
species_sizes.plot(kind="bar", color="skyblue", edgecolor="black")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Unique sequences", fontsize=12)
plt.title("Unique sequences by species", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tick_params(axis="both", which="major", labelsize=10)
plt.tight_layout()
plt.savefig(PLOT_DIR / "unique_sequences_by_species.pdf")

# Filtered label plot example
plot_filtered_labels(
    ddf,
    ["Subject", "Author"],
    "Unique_sequences",
    "Unique sequences",
    "Unique Sequences by Subject and Publication",
    PLOT_DIR / "filtered_unique_sequences_by_subject.pdf",
    label_threshold=10_000_000,
    log_scale=False,
)

plot_filtered_labels(
    ddf,
    ["Subject", "Author"],
    "Unique_sequences",
    "Unique sequences",
    "Unique Sequences by Subject and Publication",
    PLOT_DIR / "filtered_unique_sequences_by_subject2.pdf",
    label_threshold=10_000_000,
    log_scale=False,
)

