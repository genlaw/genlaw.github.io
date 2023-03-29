# Convert index.md to index.html

pandoc -s \
    --from markdown \
    --to html \
    --css pandoc.css \
    --citeproc \
    --bibliography references.bib \
    --output index.html \
    -H header.html \
    index.md


# former attempts
# pandoc -t markdown_strict --citeproc pandoc-bib-test.md -o pandoc-bib-test-output.md --bibliography references.bib
# pandoc -t html --citeproc index.md -o index.html --bibliography references.bib
# pandoc -t yaml -f bibtex references.bib
