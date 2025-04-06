#!/bin/bash

# Input parameters
input_file="$1"
training_file="$2"
validation_file="$3"
testing_file="$4"
training_fraction="$5"
validation_fraction="$6"
no_seqs_to_use="$7"
temp_location="$8"

# Calculate the testing fraction
testing_fraction=$(echo "1 - $training_fraction - $validation_fraction" | bc -l)

echo "Testing fraction: $testing_fraction"

# Extract the header (first line) from the input file
header=$(head -n 1 $input_file)

# Shuffle the data lines (skip the header) directly
tail -n +2 $input_file | shuf > "$temp_location/shuffled_data.csv"

# Limit the number of sequences to use
if [ "$no_seqs_to_use" -gt 0 ]; then
    head -n $((no_seqs_to_use + 1)) "$temp_location/shuffled_data.csv" > "$temp_location/shuffled_data_limited.csv"
    mv "$temp_location/shuffled_data_limited.csv" "$temp_location/shuffled_data.csv"
fi

# Get the total number of data lines
total_lines=$(wc -l < "$temp_location/shuffled_data.csv")

# Calculate the number of lines for training, validation, and testing
training_lines=$(echo "$total_lines * $training_fraction" | bc | awk '{print int($1)}')
validation_lines=$(echo "$total_lines * $validation_fraction" | bc | awk '{print int($1)}')
testing_lines=$(echo "$total_lines * $testing_fraction" | bc | awk '{print int($1)}')

# Save the training, validation, and testing files with the header line
echo "$header" > $training_file
head -n $training_lines "$temp_location/shuffled_data.csv" >> $training_file

# Get the remaining lines for validation
start_validation_line=$((training_lines + 1))
end_validation_line=$((training_lines + validation_lines))
echo "$header" > $validation_file
sed -n "${start_validation_line},${end_validation_line}p" "$temp_location/shuffled_data.csv" >> "$validation_file"

# Get the remaining lines for testing
start_testing_line=$((training_lines + validation_lines + 1))
echo "$header" > "$testing_file"
sed -n "${start_testing_line},\$p" "$temp_location/shuffled_data.csv" >> "$testing_file"

# Clean up the shuffled data file
#rm shuffled_data.csv
