#!/usr/bin/env sh

DATASET_LOC=${1}

for name in $DATASET_LOC*.png; do
    OUTPUT=$(identify -format '%w' $name)
    if [ $OUTPUT != "160" ]; then
        echo $name
        echo $OUTPUT
    fi 
done

