#!/bin/bash

# Ensure correct usage
if [ "$#" -ne 6 ]; then
  echo "Usage: $0 <input_file> <output_file> <nth_line> <columns> <download_dir> <progress_file>"
  exit 1
fi

# Parameters
input_file="$1"
output_file="$2"
nth_line="$3"
columns="$4"
download_dir="$5"
progress_file="$6"

echo "Input file: $input_file"
echo "Output file: $output_file"
echo "Every nth line: $nth_line"
echo "Columns to keep: $columns"
echo "Progress tracking file: $progress_file"

module load eth_proxy

# Create and use a dedicated 'downloading' folder
mkdir -p "$download_dir"

# Initialize output file with header only if it doesn't exist or is empty
if [ ! -s "$output_file" ]; then
  # The first row is the new header with Sequence_ID and File_ID added
  (echo "Sequence_ID,File_ID"; csvcut -c "$columns" "assets/header.csv" | head -n 1) | paste -sd ',' > "$output_file"
  echo "Header written to $output_file"
fi

# Initialize or read progress tracking file
if [ ! -f "$progress_file" ]; then
  touch "$progress_file"
fi

# Initialize sequence counter - start from the highest ID in existing file or from 1
if [ -s "$output_file" ]; then
  # Get the highest sequence ID from the existing file (skip header)
  last_seq=$(tail -n +2 "$output_file" | cut -d ',' -f 1 | sort -n | tail -1)
  if [ -z "$last_seq" ]; then
    seq_counter=1
  else
    seq_counter=$((last_seq + 1))
  fi
else
  seq_counter=1
fi

# Process each file in the input list
total_lines=$(wc -l < "$input_file")
current_line=0

while IFS=, read -r file_id url; do
  # Skip header line
  if [[ "$file_id" == "file_id" ]]; then
    continue
  fi
  
  current_line=$((current_line + 1))
  echo "Processing $current_line of $total_lines: $file_id"
  
  # Check if file was already processed
  if grep -q "^$file_id\$" "$progress_file"; then
    echo "File $file_id already processed, skipping..."
    continue
  fi
  
  # Create a temporary file
  temp_file="$download_dir/temp_$file_id.csv"
  
  # Download and process the file
  echo "Downloading $url"
  
  # Create temp directory
  temp_dir="$download_dir/tmp/$file_id"
  mkdir -p "$temp_dir"
  
  # Download the file
  if ! curl -L "$url" -o "$temp_dir/file.gz"; then
    echo "Failed to download $url, continuing with next file..."
    continue
  fi
  
  # Unzip the file
  if ! gunzip -f "$temp_dir/file.gz"; then
    echo "Failed to unzip file for $file_id, continuing with next file..."
    rm -rf "$temp_dir"
    continue
  fi
  
  # Remove first two rows from original file (in-place) - THIS WAS MISSING
  sed -i '1,2d' "$temp_dir/file"
  
  # Filter the file and add to output
  echo "Filtering file..."
  csvgrep -c 35 -r '^(?!\s*$).{20,50}$' "$temp_dir/file" | \
  csvgrep -c 45 -r '^(?!\s*$).{10,50}$' | \
  csvgrep -c 37 -r '^(?!\s*$).{5,12}$' | \
  csvgrep -c 41 -r '^(?!\s*$).{1,10}$' | \
  csvgrep -c 47 -r '^(?!\s*$).{5,38}$' > "$temp_dir/filtered.csv"
  
  # If filtered file is not empty, add to output
  if [ -s "$temp_dir/filtered.csv" ]; then
    echo "Extracting columns: $columns"
    
    # Process every nth line from the filtered file (matching download.sh behavior)
    awk -v n="$nth_line" -v seq="$seq_counter" -v fid="$file_id" '
    NR > 1 && NR % n == 0 {  # Skip header line and take every nth line
      print seq "," fid "," $0;
      seq++;
    }' <(csvcut -c "$columns" "$temp_dir/filtered.csv") > "$temp_file"
    
    # Append to output
    cat "$temp_file" >> "$output_file"
    
    # Update sequence counter
    if [ -s "$temp_file" ]; then
      new_seqs=$(wc -l < "$temp_file")
      seq_counter=$((seq_counter + new_seqs))
      echo "Added $new_seqs sequences from $file_id"
    else
      echo "No sequences matched the nth line criteria from $file_id"
    fi
    
    # Mark file as processed
    echo "$file_id" >> "$progress_file"
  else
    echo "No sequences passed filtering for $file_id"
    # Still mark as processed
    echo "$file_id" >> "$progress_file"
  fi
  
  # Clean up
  rm -f "$temp_file"
  rm -rf "$temp_dir"
  
  echo "Completed processing $file_id"
  
done < "$input_file"

echo "Download process complete!"
echo "Data has been written to $output_file"