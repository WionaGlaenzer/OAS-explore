#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_csv> <output_csv> <number_of_samples>"
    echo "Example: $0 input.csv output.csv 100"
    exit 1
fi

input_csv="$1"
output_csv="$2"
n_samples="$3"

# Get total number of lines (excluding header)
total_lines=$(wc -l < "$input_csv")
total_sequences=$((total_lines - 1))

if [ "$n_samples" -gt "$total_sequences" ]; then
    echo "Error: Requested sample size ($n_samples) is larger than total sequences ($total_sequences)"
    exit 1
fi

# Write header to output file
head -n 1 "$input_csv" > "$output_csv"

# Sample n random lines (excluding header) and append to output
#tail -n +2 "$input_csv" | shuf -n "$n_samples" >> "$output_csv"
seed=42                                 # pick any integer or string
tail -n +2 "$input_csv" \
  | shuf -n "$n_samples" \
        --random-source=<(openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null) \
  >> "$output_csv"

echo "Successfully sampled $n_samples sequences from $input_csv to $output_csv" 