#!/usr/bin/env bash
# Download the four public MoleculeNet-format CSV files used by the benchmark into ./data_raw
set -euo pipefail
mkdir -p data_raw
BASE="https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest"
for f in tox21.csv clintox.csv sider.csv toxcast_data.csv; do
  curl -fsSL "$BASE/$f" -o "data_raw/$f"; echo "downloaded $f"
done
