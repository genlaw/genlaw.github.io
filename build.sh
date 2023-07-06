# `./build.sh` will build all markdown files in the directory.
# Alternatively, use `./build.sh filename` to build a specific file. 

if [ $# -eq 0 ];
then
    files=( $(find "." -maxdepth 2 -name '*.md' ! -name 'README.md') )
    # Print out the filenames in the array
else
    files=( $1 )
fi

for file in "${files[@]}"
do
    filename="${file%.*}"
    echo "Compiling $filename"

    DIR=$(dirname "$file")

    if [[ "$DIR" != "." ]]; then
        STYLE_FILE="./../styles.css"
    else
        STYLE_FILE="styles.css"
    fi

    pandoc -s \
        --from markdown \
        --to html \
        --wrap none \
        --css "$STYLE_FILE" \
        --citeproc \
        --toc \
        --bibliography references.bib \
        --template template.html \
        --output $filename.html \
        $filename.md
done


# former attempts
# pandoc -t markdown_strict --citeproc pandoc-bib-test.md -o pandoc-bib-test-output.md --bibliography references.bib
# pandoc -t html --citeproc index.md -o index.html --bibliography references.bib
# pandoc -t yaml -f bibtex references.bib
