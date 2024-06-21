#!/bin/bash

# Check if calcmethod is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <calcmethod>"
    exit 1
fi

# Assign the first argument to the calcmethod variable
calcmethod=$1

# Loop through all items in the current directory
for dir in *; do
    # Check if the item is a directory
    if [ -d "$dir" ]; then
        echo $dir $calcmethod
        python train.py -p $dir/$calcmethod
    fi
done
