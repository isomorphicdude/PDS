#!/bin/bash

# Intended for Colab Notebooks

# Define variables
REPO_URL= "https://github.com/isomorphicdude/PDS.git"
REPO_DIR="/content/PDS"
OUTPUT_FILE="output.txt"


ARG1 = "--compute_fid"
ARG2 = "--config /content/PDS/configs/ve/cifar10_ncsnpp_deep_continuous.py"
ARG3 = "--workdir /content/PDS/"
ARG4 = "--speed_up 20"
ARG5 = "--verbose"

# Clone the repository
git clone "$REPO_URL" "$REPO_DIR"

# Change to the repository directory
cd "$REPO_DIR"

# Install Python packages from requirements.txt
pip install -r requirements.txt

# Download data by running a Python script without input
python download_ckpt.py

# Run a Python script with arguments arg1, arg2, arg3 and save output
python main.py "$ARG1" "$ARG2" "$ARG3" "$ARG4" > "$OUTPUT_FILE" 2>&1

# Optionally, you can also append the output to the file instead of overwriting
# python your_script.py "$ARG1" "$ARG2" "$ARG3" >> "$OUTPUT_FILE" 2>&1
