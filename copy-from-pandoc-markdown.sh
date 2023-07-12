#!/bin/bash

source_dir="/Users/katherine/Projects/pandoc-markdown-css-theme/docs/training-data/"
destination_dir="/Users/katherine/Projects/genlaw.github.io/"
file_list=("resources" "glossary" "papers" "cfp")

for filename in "${file_list[@]}"
do
    source_path="${source_dir}${filename}.html"
    destination_path="${destination_dir}${filename}.html"

    cp "$source_path" "$destination_path" && echo "Copied: $filename"
done


source_dir="/Users/katherine/Projects/pandoc-markdown-css-theme/docs/training-data/"
destination_dir="/Users/katherine/Projects/genlaw.github.io/explainers/"
file_list=("index" "training-data")

for filename in "${file_list[@]}"
do
    source_path="${source_dir}${filename}.html"
    destination_path="${destination_dir}${filename}.html"

    cp "$source_path" "$destination_path" && echo "Copied: $filename"
done

# Copy CSS
source_dir="/Users/katherine/Projects/pandoc-markdown-css-theme/docs/training-data/css/"
destination_dir="/Users/katherine/Projects/genlaw.github.io/css/"
file_list=("theme" "theme-additions")

for filename in "${file_list[@]}"
do
    source_path="${source_dir}${filename}.css"
    destination_path="${destination_dir}${filename}.css"

    cp "$source_path" "$destination_path" && echo "Copied: $filename"
done

# Copy template
source_path="/Users/katherine/Projects/pandoc-markdown-css-theme/template.html5"
destination_path="/Users/katherine/Projects/genlaw.github.io/template.html5"
cp "$source_path" "$destination_path" && echo "Copied: template.html5"

