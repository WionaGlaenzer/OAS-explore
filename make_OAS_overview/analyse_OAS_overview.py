import pandas as pd
import matplotlib.pyplot as plt

# Specify the input CSV file name
input_file = "OAS_overview.csv"  # Replace with your actual CSV file

# Read the CSV file into a pandas DataFrame
print("starting to read in")
df = pd.read_csv(input_file, sep=",")
print("read in the file")

# Sum the "size_MB" column (make sure it's numeric)
total_size = df["Size_MB"].sum()

# Print the total size
print(f"Total size in MB: {total_size}")

# Covert to GB and print
total_size_gb = total_size / 1024
print(f"Total size in GB: {total_size_gb}")

# Convert to TB and print
total_size_tb = total_size_gb / 1024
print(f"Total size in TB: {total_size_tb}")

def plot_grouped_data(df, group_columns, sum_column, y_label, title, file_name, log_scale=False):
    """
    Function to group data by specified columns, sum a specific column, and plot a bar chart.
    
    Parameters:
    - df: DataFrame to be used for plotting
    - group_columns: Columns to group by (can be one or more)
    - sum_column: The column to sum after grouping
    - y_label: Label for the y-axis
    - title: Title for the plot
    - file_name: Path to save the plot
    - log_scale: Whether to use a logarithmic scale on the y-axis (default is False)
    """
    # Group by the specified columns and sum the specified column
    grouped_data = df.groupby(group_columns)[sum_column].sum()
    
    # Plot the data
    plt.figure(figsize=(20, 6))
    grouped_data.plot(kind="bar", color='skyblue', edgecolor='black')
    
    # Set logarithmic scale if specified
    if log_scale:
        plt.yscale('log')
    
    # Customize the plot
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(file_name)
    plt.close()

# Plot total size by publication
plot_grouped_data(df, ["Author"], "Size_MB", "Total Size (MB)", "Total Size by Publication", "plots/total_size_by_publication.pdf")

# Plot total sequences by publication
plot_grouped_data(df, ["Author"], "Total sequences", "Total sequences", "Total sequences by Publication", "plots/total_sequences_by_publication.pdf")

# Plot unique sequences by publication
plot_grouped_data(df, ["Author"], "Unique sequences", "Unique sequences", "Unique sequences by Publication", "plots/unique_sequences_by_publication.pdf")

# Plot unique sequences by subject and publication
plot_grouped_data(df, ["Subject", "Author"], "Unique sequences", "Unique sequences", "Unique Sequences by Subject and Publication", "plots/unique_sequences_by_subject_log.pdf", log_scale=True)

plot_grouped_data(df, ["Subject", "Author"], "Unique sequences", "Unique sequences", "Unique Sequences by Subject and Publication", "plots/unique_sequences_by_subject.pdf")


# Histogram of subjects within a range of unique sequences per subject
subject_sizes = df.groupby(["Subject", "Author"])["Unique sequences"].sum()
subject_sizes = subject_sizes.reset_index()
num_bins = 10
subject_sizes_bins = pd.cut(subject_sizes["Unique sequences"], bins=num_bins)
bin_edges = [0, 100, 1000, 10000, 100000, 1000000, 10000000,float('inf')]  # Customize this based on your data
subject_sizes_bins = pd.cut(subject_sizes["Unique sequences"], bins=bin_edges)
bin_counts = subject_sizes_bins.value_counts().sort_index()
plt.figure(figsize=(10, 6))
bin_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel("Range of Unique Sequences per Individual", fontsize=12)
plt.ylabel("Number of Subjects", fontsize=12)
plt.title("Histogram of Subjects by Range of Unique Sequences", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.savefig("plots/histogram_unique_sequences_by_subject.pdf")

# Plot the number of sequences per species
species_sizes = df.groupby("Species")["Unique sequences"].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
species_sizes.plot(kind="bar", color='skyblue', edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Unique sequences", fontsize=12)
plt.title("Unique sequences by species", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.savefig("plots/unique_sequences_by_species.pdf")

def plot_filtered_labels(df, group_columns, sum_column, y_label, title, file_name, label_threshold, log_scale=False):
    """
    Groups data by specified columns, sums a specific column, and plots a bar chart,
    but only shows x-axis labels for entries above a given threshold (keeping original order).

    Parameters:
    - df: DataFrame to be used for plotting
    - group_columns: Columns to group by (one or more)
    - sum_column: The column to sum after grouping
    - y_label: Label for the y-axis
    - title: Title for the plot
    - file_name: Path to save the plot
    - label_threshold: Minimum value of sum_column required to display the x-axis label
    - log_scale: Whether to use a logarithmic scale on the y-axis (default: False)
    """
    # Group the data while preserving the original order
    grouped_data = df.groupby(group_columns, sort=False)[sum_column].sum()

    # Create labels: Keep only those above threshold, replace others with empty strings
    labels = [idx if value > label_threshold else "" for idx, value in zip(grouped_data.index, grouped_data.values)]
    
    # Plot the data
    plt.figure(figsize=(20, 12))
    grouped_data.plot(kind="bar", color='skyblue', edgecolor='black')

    # Set custom x-tick labels with filtered labels (preserving order)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha='right')

    # Set logarithmic scale if needed
    if log_scale:
        plt.yscale('log')

    # Customize the plot
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()

    # Save the plot
    plt.savefig(file_name)
    plt.close()

# Example usage with a threshold of 1,000,000 unique sequences
plot_filtered_labels(df, ["Subject", "Author"], "Unique sequences", 
                     "Unique sequences", "Unique Sequences by Subject and Publication",
                     "plots/filtered_unique_sequences_by_subject.pdf",
                     label_threshold=10_000_000, log_scale=False)

def plot_filtered_labels(df, group_columns, sum_column, y_label, title, file_name, label_threshold, log_scale=False):
    """
    Groups data by specified columns, sums a specific column, and plots a bar chart,
    but only shows x-axis labels for entries above a given threshold (keeping original order).

    Parameters:
    - df: DataFrame to be used for plotting
    - group_columns: Columns to group by (one or more)
    - sum_column: The column to sum after grouping
    - y_label: Label for the y-axis
    - title: Title for the plot
    - file_name: Path to save the plot
    - label_threshold: Minimum value of sum_column required to display the x-axis label
    - log_scale: Whether to use a logarithmic scale on the y-axis (default: False)
    """
    # Group the data while preserving the original order
    grouped_data = df.groupby(group_columns, sort=False)[sum_column].sum()

    # Create labels: Keep only those above threshold, replace others with empty strings
    labels = [idx if value > label_threshold else "" for idx, value in zip(grouped_data.index, grouped_data.values)]
    
    # Plot the data
    plt.figure(figsize=(20, 16))
    bars = plt.bar(range(len(grouped_data)), grouped_data.values, color='skyblue', edgecolor='black')

    # Set custom x-tick labels with filtered labels (preserving order)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha='right', fontsize=10)

    # Set logarithmic scale if needed
    if log_scale:
        plt.yscale('log')

    # Customize the plot
    plt.ylabel(y_label, fontsize=14)
    plt.title(title, fontsize=16, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Annotate bars with their values if they are above a certain threshold
    for bar in bars:
        height = bar.get_height()
        if height > label_threshold:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', 
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save the plot
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

# Example usage with a threshold of 1,000,000 unique sequences
plot_filtered_labels(df, ["Subject", "Author"], "Unique sequences", 
                     "Unique sequences", "Unique Sequences by Subject and Publication",
                     "plots/filtered_unique_sequences_by_subject2.pdf",
                     label_threshold=10_000_000, log_scale=False)