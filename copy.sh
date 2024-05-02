#!/bin/bash

# Copy HTML
source_dir="/Users/katherine/Projects/pandoc-markdown-css-theme/docs/training-data/"
destination_dir="/Users/katherine/Projects/genlaw.github.io/"
file_list=("resources" "glossary" "papers" "cfp" "2023-full-report" "2023-report" "metaphors" "index")

for filename in "${file_list[@]}"
do
    source_path="${source_dir}${filename}.html"
    destination_path="${destination_dir}${filename}.html"

    cp "$source_path" "$destination_path" && echo "Copied: $filename.html"
done


source_dir="/Users/katherine/Projects/pandoc-markdown-css-theme/docs/training-data/explainers/"
destination_dir="/Users/katherine/Projects/genlaw.github.io/explainers/"

for file in "$source_dir"/*;
do
    filename=$(basename "$file")
    source_path="${source_dir}${filename}"
    destination_path="${destination_dir}${filename}"

    cp -r "$source_path" "$destination_path" && echo "Copied: $filename"
done

# Copy md
source_dir="/Users/katherine/Projects/pandoc-markdown-css-theme/src/training-data/"
destination_dir="/Users/katherine/Projects/genlaw.github.io/"
file_list=("resources" "glossary" "papers" "cfp" "2023-full-report" "2023-report" "metaphors" "index")

for filename in "${file_list[@]}"
do
    source_path="${source_dir}${filename}.md"
    destination_path="${destination_dir}${filename}.md"

    cp -r "$source_path" "$destination_path" && echo "Copied: $filename.md"
done


source_dir="/Users/katherine/Projects/pandoc-markdown-css-theme/src/training-data/explainers/"
destination_dir="/Users/katherine/Projects/genlaw.github.io/explainers/"

for file in "$source_dir"/*;
do
    filename=$(basename "$file")
    source_path="${source_dir}${filename}"
    destination_path="${destination_dir}${filename}"

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

# Copy references
source_path="/Users/katherine/Projects/pandoc-markdown-css-theme/src/training-data/references.bib"
destination_path="/Users/katherine/Projects/genlaw.github.io/references.bib"
cp "$source_path" "$destination_path" && echo "Copied: references.bib"

## GenLaw 2024 ICML

source_path="/Users/katherine/Projects/pandoc-markdown-css-theme/docs/training-data/2024-icml-cfp.html"
destination_path="/Users/katherine/Projects/genlaw.github.io/2024-icml/cfp.html"
cp "$source_path" "$destination_path" && echo "Copied: 2024-icml-cfp.html"

source_path="/Users/katherine/Projects/pandoc-markdown-css-theme/src/training-data/2024-icml-cfp.md"
destination_path="/Users/katherine/Projects/genlaw.github.io/2024-icml/cfp.md"
cp "$source_path" "$destination_path" && echo "Copied: 2024-icml-cfp.md"