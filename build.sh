# pandoc -t markdown_strict --citeproc pandoc-bib-test.md -o pandoc-bib-test-output.md --bibliography references.bib
# pandoc -t html --citeproc index.md -o index.html --bibliography references.bib

pandoc -s -f markdown -t html --css pandoc.css index.md -o index.html --citeproc --bibliography references.bib index.md 


# pandoc -t yaml -f bibtex references.bib
