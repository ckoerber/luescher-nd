
ALL=FV_corrections.pdf luescher-logic.pdf notes.pdf towerOps.pdf

all: $(ALL)

%.pdf: %.tex
	pdflatex $*
	-bibtex $*
	pdflatex $*
	pdflatex $*

tidy:
	$(RM) *.{aux,bbl,blg,log}

