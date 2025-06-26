#!/bin/bash
export TMPDIR=/REDACTED/PATHroject/reddy/REDACTED/PATHr -p $TMPDIR

# Input parameters
input_file="$1"
training_file="$2"
validation_file="$3"
testing_file="$4"
training_fraction="$5"
validation_fraction="$6"
no_seqs_to_use="$7"
temp_location="$8"
split_mode="${9:-fraction}"  # New parameter: "fraction" or "numbers"

# Derive .csv filenames from .txt filenames
training_csv="${training_file%.txt}.csv"
validation_csv="${validation_file%.txt}.csv"
testing_csv="${testing_file%.txt}.csv"

# Create temp directory if it doesn't exist
mkdir -p "$temp_location"

# Extract header
header=$(head -n 1 "$input_file")

# Shuffle data (excluding header)
tail -n +2 "$input_file" | shuf > "$temp_location/shuffled_data.csv"

# Limit number of sequences if specified
if [ "$no_seqs_to_use" -gt 0 ]; then
    head -n $no_seqs_to_use "$temp_location/shuffled_data.csv" > "$temp_location/shuffled_data_limited.csv"
    mv "$temp_location/shuffled_data_limited.csv" "$temp_location/shuffled_data.csv"
fi

# Count total data lines
total_lines=$(wc -l < "$temp_location/shuffled_data.csv")

# Calculate splits based on mode
if [ "$split_mode" = "numbers" ]; then
    # Mode: Use raw numbers
    training_lines="$training_fraction"
    validation_lines="$validation_fraction"
    testing_lines=$((total_lines - training_lines - validation_lines))
    
    # Validate that numbers don't exceed total
    if [ $((training_lines + validation_lines)) -gt $total_lines ]; then
        echo "Error: Training + validation sequences ($((training_lines + validation_lines))) exceed total available ($total_lines)"
        exit 1
    fi
    
    echo "Using raw numbers:"
    echo "Training sequences: $training_lines"
    echo "Validation sequences: $validation_lines"
    echo "Testing sequences: $testing_lines"
    echo "Total sequences: $total_lines"
    
else
    # Mode: Use fractions (default behavior)
    testing_fraction=$(echo "1 - $training_fraction - $validation_fraction" | bc -l)
    echo "Testing fraction: $testing_fraction"
    
    # Calculate splits from fractions
    training_lines=$(echo "$total_lines * $training_fraction" | bc | awk '{print int($1)}')
    validation_lines=$(echo "$total_lines * $validation_fraction" | bc | awk '{print int($1)}')
    testing_lines=$((total_lines - training_lines - validation_lines))
    
    echo "Using fractions:"
    echo "Training fraction: $training_fraction -> $training_lines sequences"
    echo "Validation fraction: $validation_fraction -> $validation_lines sequences"
    echo "Testing fraction: $testing_fraction -> $testing_lines sequences"
    echo "Total sequences: $total_lines"
fi

# Generate training CSV
echo "$header" > "$training_csv"
head -n "$training_lines" "$temp_location/shuffled_data.csv" >> "$training_csv"

# Generate validation CSV
start_validation_line=$((training_lines + 1))
end_validation_line=$((training_lines + validation_lines))
echo "$header" > "$validation_csv"
sed -n "${start_validation_line},${end_validation_line}p" "$temp_location/shuffled_data.csv" >> "$validation_csv"

# Generate testing CSV
start_testing_line=$((training_lines + validation_lines + 1))
echo "$header" > "$testing_csv"
sed -n "${start_testing_line},\$p" "$temp_location/shuffled_data.csv" >> "$testing_csv"

# Generate .txt files with concatenated columns 4–10 (no header)
tail -n +2 "$training_csv" | awk -F',' '{OFS=""; for(i=4;i<=10;i++) printf $i; print ""}' > "$training_file"
tail -n +2 "$validation_csv" | awk -F',' '{OFS=""; for(i=4;i<=10;i++) printf $i; print ""}' > "$validation_file"
tail -n +2 "$testing_csv" | awk -F',' '{OFS=""; for(i=4;i<=10;i++) printf $i; print ""}' > "$testing_file"

# Done!
echo "Generated:"
echo "CSV files: $training_csv, $validation_csv, $testing_csv (with headers)"
echo "TXT files: $training_file, $validation_file, $testing_file (no headers, columns 4 to 10 concatenated)"