"""
download_data.py
Utility script to download the canonical 1,338-record insurance.csv dataset.
"""

import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "insurance.csv")

if __name__ == "__main__":
    print(f"Downloading official insurance dataset from: {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, TARGET)
    print(f"Dataset successfully saved to: {TARGET} ({os.path.getsize(TARGET):,} bytes)")
