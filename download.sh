#!/bin/bash

# Ensure proper usage
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_file> <output_file> <nth_line> <columns>"
    exit 1
fi

# Parameters
input_file="$1"
output_file="$2"
nth_line="$3"
columns="$4"

# Create and use a dedicated 'downloading' folder
download_dir="downloading"
mkdir -p "$download_dir"

echo "Input file: $input_file"
echo "Output file: $output_file"
echo "Every n-th line: $nth_line"
echo "Columns to keep: $columns"

# Clear the output file before writing new results
> "$output_file"

count=1

while read -r url; do 
    if (( count % nth_line == 0 )); then
        echo "Downloading: $url"
        
        # Download file into 'downloading' directory
        wget -P "$download_dir" "$url"

        # Unzip any .gz files
        echo "Unzipping files..."
        for file in "$download_dir"/*.gz; do
            if [ -f "$file" ]; then
                gunzip "$file"
            fi
        done

        # Process CSV files
        for file in "$download_dir"/*.csv; do
            if [ -r "$file" ]; then
                echo "Processing file: $(basename "$file")"
                
                awk -v cols="$columns" '
                BEGIN {
                    FS = ","  # Standard CSV delimiter
                    OFS = ","  # Ensure output uses commas as well
                    split(cols, colArr, ",")  # Convert user-specified columns into an array
                }
                function clean_field(field) {
                    gsub(/^"|"$/, "", field)  # Remove leading/trailing quotes
                    return field
                }
                NR > 2 && length($35) >= 20 && length($45) >= 10 && length($37) >= 5 && length($37) <= 12 && length($41) <= 10 && length($41) >= 1 && length($47) >= 5 && length($47) <= 38 {
                    out = ""
                    for (i in colArr) {
                        field = clean_field($colArr[i])  # Clean up quotes
                        if (out == "") {
                            out = field  # First column (no leading comma)
                        } else {
                            out = out OFS field  # Append subsequent columns with commas
                        }
                    }
                    print out
                }' "$file" >> "$output_file"
            fi
        done

        # Clean up downloaded files
        rm -f "$download_dir"/*.csv
        rm -f "$download_dir"/*.gz
    fi
    ((count++))
done < "$input_file"

echo "Data has been written to $output_file."
