#
# Makefile for building the report
#

MAIN_FILE='main'

build: clean
	@pdflatex -synctex=1 -interaction=nonstopmode $(MAIN_FILE).tex > /dev/null
	@bibtex $(MAIN_FILE).aux > /dev/null
	@pdflatex -synctex=1 -interaction=nonstopmode $(MAIN_FILE).tex > /dev/null
	@pdflatex -synctex=1 -interaction=nonstopmode $(MAIN_FILE).tex > /dev/null
	echo 'Created report and saved to pdf'

clean:
	@rm -f *blx.bib *.out *.toc *.log *.pdf *.xml *.lot *.glsdefs \
				 *.aux *.ist *.lof *.synctex.gz *.bbl *.blg *.glo
	@echo 'Removed generated files'

.PHONY: build clean