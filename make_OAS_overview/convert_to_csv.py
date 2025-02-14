import csv
import json
import re

# Input and output file names
input_file = "OAS_overview.txt"   # Replace with your actual input file name
output_file = "OAS_overview.csv"  # Desired CSV output file

# Open the input and output files
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = None  # CSV writer (to be initialized later)

    for line in infile:
        # Match pattern using regex (improved for JSON parsing)
        match = re.match(r'wget (\S+), "(.*?)", ([\d.]+) MB$', line.strip())
        if not match:
            print(f"Skipping invalid line: {line.strip()}")
            continue

        url, metadata_json, size_mb = match.groups()  # Extract components

        # Fix improperly escaped JSON (convert `""` → `"`)
        metadata_json = metadata_json.replace('""', '"')

        try:
            # Convert metadata from JSON-like string to dictionary
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            print(f"Skipping malformed JSON: {metadata_json}")
            continue

        # Ensure writer is initialized with header
        if writer is None:
            fieldnames = ["Download_Link"] + list(metadata.keys()) + ["Size_MB"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

        # Write the extracted data to CSV
        row_data = {"Download_Link": url, **metadata, "Size_MB": size_mb}
        writer.writerow(row_data)

print(f"CSV file '{output_file}' created successfully!")
