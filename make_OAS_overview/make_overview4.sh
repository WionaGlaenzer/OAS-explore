#!/bin/bash

# Load the proxy module (if needed)
module load eth_proxy

# Output file to store the first row and file size
output_file="OAS_overview.txt"

# Ensure the output file is empty before starting
> "$output_file"

# File containing the list of wget commands
url_file="bulk_download.sh"

# Create a temporary directory for downloaded files
temp_dir=$(mktemp -d)
trap 'rm -rf -- "$temp_dir"' EXIT  # Clean up temporary directory on exit

# Read each line from the file and execute it as a Bash command
while IFS= read -r command; do
    # Modify the command to suppress wget's output (-q for quiet mode)
    silent_command=$(echo "$command" | sed 's/wget /wget -q /')

    # Execute the wget command (download the file into the temporary directory)
    echo "Downloading: $(echo "$command" | awk '{print $2}')"
    eval "$silent_command -P $temp_dir"  # -P specifies the output directory

    # Extract the filename from the wget command
    filename=$(basename "$(echo "$command" | awk '{print $2}')")

    # Unzip the file
    csv_file="${temp_dir}/${filename%.gz}"
    echo "Unzipping $filename..."
    gunzip -f "$temp_dir/$filename"

    # Extract the first row of the CSV file
    first_row=$(head -n 1 "$csv_file")

    # Get the size of the unzipped CSV file in bytes
    file_size_bytes=$(stat -c%s "$csv_file")

    # Convert file size from bytes to MB (1024^2 = 1048576) and format it to 4 decimal places
    file_size_mb=$(echo "scale=6; $file_size_bytes / 1048576" | bc)  # Convert with high precision
    file_size_mb=$(printf "%.4f" "$file_size_mb")  # Format to 4 decimal places

    # Append the URL (wget command), first row, and file size (in MB) to output.txt
    echo "$command, $first_row, $file_size_mb MB" >> "$output_file"

    # Clean up the extracted CSV file
    rm "$csv_file"

    echo "Processed $filename successfully."
done < "$url_file"

echo "All files processed. Results saved in $output_file."
