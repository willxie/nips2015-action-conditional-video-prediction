#!/usr/bin/env sh

INPUT_PATH=$1

for name in $INPUT_PATH*.png; do
    convert -resize 160x210\! $name $name
done

echo "Done."
