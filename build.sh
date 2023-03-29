# Convert index.md to index.html

if [ $# -eq 0 ];
then
    files=( $(find "." -maxdepth 1 -name '*.md' ! -name 'README.md') )
    # Print out the filenames in the array
else
    files=( $1 )
fi

for file in "${files[@]}"
do
    filename="${file%.*}"
    echo "Compiling $filename"
    
    pandoc -s \
        --from markdown \
        --to html \
        --css pandoc.css \
        --citeproc \
        --bibliography references.bib \
        --output $filename.html \
        -H header.html \
        $filename.md
done


# former attempts
# pandoc -t markdown_strict --citeproc pandoc-bib-test.md -o pandoc-bib-test-output.md --bibliography references.bib
# pandoc -t html --citeproc index.md -o index.html --bibliography references.bib
# pandoc -t yaml -f bibtex references.bib
