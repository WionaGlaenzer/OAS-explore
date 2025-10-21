import pandas as pd
import os, sys
from pipeline.data_functions import csv_to_txt

def main():
    # File paths
    folder = "/REDACTED/PATH"

    # Transform all files from the folder to txt with the same name
    for file in os.listdir(folder):
        print(file)
        if file.endswith(".csv"):
            csv_path = os.path.join(folder, file)
            txt_path = os.path.join(folder, file.replace(".csv", ".txt"))
            print(f"Converting {csv_path} to {txt_path}")
            csv_to_txt(csv_path, txt_path)
            print(f"Converted {csv_path} to {txt_path}")

if __name__ == "__main__":
    main()