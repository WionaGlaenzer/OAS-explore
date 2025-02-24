#!/bin/bash
# Ensure correct usage
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <input_file> <output_file> <nth_line> <columns> <download_dir>"
  exit 1
fi

# Parameters
input_file="$1"
output_file="$2"
nth_line="$3"
columns="$4"
download_dir="$5"

# Create and use a dedicated 'downloading' folder
mkdir -p "$download_dir"
echo "Input file: $input_file"
echo "Output file: $output_file"
echo "Every n-th line: $nth_line"
echo "Columns to keep: $columns"

module load eth_proxy

# Clear the output file before writing new results
> "$output_file"
count=1
header_written=false # Track if header has been written

# Add a sequence counter variable
seq_counter=1

# Skip the header line and then read the rest of the file
tail -n +2 "$input_file" | while IFS=',' read -r file_id url; do
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
  
  # Process CSV files using csvkit
  for file in "$download_dir"/*.csv; do
    if [ -r "$file" ]; then
      echo "Processing file: $(basename "$file")"
      
      # Save first two rows to a separate file
      temp_file_header="${file}.header.csv"
      head -n 2 "$file" > "$temp_file_header"
      
      # Remove first two rows from original file (in-place)
      sed -i '1,2d' "$file"
      
      # Process every nth line from the file
      count_in_file=1  # Line counter within the current file
      while IFS= read -r line; do
        if (( count_in_file % nth_line == 0 )); then
          # Save filtered results to a temporary file
          temp_file="${file}.filtered.csv"
          csvgrep -c 35 -r '.{20,}' "$file" | \
          csvgrep -c 45 -r '.{10,}' | \
          csvgrep -c 37 -r '^.{5,12}$' | \
          csvgrep -c 41 -r '^.{1,10}$' | \
          csvgrep -c 47 -r '^.{5,38}$' > "$temp_file"

          # If the filtered file is not empty, extract columns
          if [ -s "$temp_file" ]; then
            echo "Extracting columns: $columns"
            if [ "$header_written" = false ]; then
              echo "Writing header..."
              # Add Sequence_ID to header
              (echo "Sequence_ID,File_ID"; csvcut -c "$columns" "assets/header.csv" | head -n 1) | paste -sd ',' > "$output_file"
              header_written=true
            fi
            # Append filtered data with Sequence_ID and File_ID
            while IFS= read -r line; do
              echo "$seq_counter,$file_id,$line" >> "$output_file"
              ((seq_counter++))
            done < <(csvcut -c "$columns" "$temp_file")
          fi
          echo "Done processing $(basename "$file")."
          # Clean up temporary files
          rm -f "$temp_file_header"
          rm -f "$temp_file"
        fi
        ((count_in_file++))
      done < "$file"
      
      # Clean up downloaded files
      rm -f "$download_dir"/*.csv
      rm -f "$download_dir"/*.gz
    fi
  done

  ((count++))
done

echo "Data has been written to $output_file."
